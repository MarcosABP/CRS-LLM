import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, precision_score, recall_score

from libs.extract import fuzzy_match, get_ground_truth, create_binary_vectors


# CERTO, NAO MUDAR
def recall_at_k(hits, num_gt, k):
    hits_at_k = hits[:k]
    return sum(hits_at_k) / num_gt

# CERTO, NAO MUDAR
def dcg_at_k(hits, k):
    if len(hits) == 1:
        return hits[0]
    k = min(k, len(hits))
    return hits[0] + sum(hits[i] / np.log2(i + 2) for i in range(1, k))

# CERTO, NAO MUDAR
def ndcg_at_k(hits, num_gt, k):
    idea_hits = np.zeros(len(hits), dtype=int)
    idea_hits[:num_gt] = 1
    idea_dcg = dcg_at_k(idea_hits, k)
    dcg = dcg_at_k(hits, k)
    return dcg/idea_dcg


# NAO SEI, VER
def comparar_orig_extracao(test_data, responses_list):
    """
    Calcula Jaccard, Precision e Recall comparando a extração da LLM com o Gabarito.
    """
    
    # Dicionário para guardar as listas de scores
    results = {
        'jaccard': [],
        'precision': [],
        'recall': []
    }
    
    for conv, llm_text in zip(test_data, responses_list):
        
        # 1. Definição do Universo (Features)
        mentions = conv.get('movieMentions', {})
        if not mentions:
            # Se não há filmes na conversa, consideramos tudo 0
            for k in results: results[k].append(0.0)
            continue
            
        universe_ids = list(mentions.keys())
        
        # 2. Extração dos Sets (Gabarito vs Predição)
        # Nota: Certifique-se que suas funções auxiliares (get_ground_truth_ids e fuzzy_match)
        # estão definidas no seu notebook conforme passo anterior.
        true_ids_set = get_ground_truth(conv)       
        pred_ids_set = fuzzy_match(llm_text, mentions)
        
        # 3. Vetorização (y_true e y_pred)
        # y_true = Gabarito (O que o usuário realmente gostou)
        # y_pred = LLM (O que o modelo disse que ele gostou)
        y_true, y_pred = create_binary_vectors(true_ids_set, pred_ids_set, universe_ids)
        
        # 4. Cálculo das Métricas
        # zero_division=0: Evita erro se o denominador for 0 (retorna 0.0)
        
        # Jaccard (Interseção / União)
        jac = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Precision (Acertos / Total Predito) -> "O quanto a LLM alucinou?"
        prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Recall (Acertos / Total Real) -> "O quanto a LLM esqueceu?"
        rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
        
        results['jaccard'].append(jac)
        results['precision'].append(prec)
        results['recall'].append(rec)

    # Calcula as médias
    means = {k: np.mean(v) for k, v in results.items()}
    
    return means, results



# NAO SEI, VER
def avaliar_orig(dataset, ease_model, movie_mapper, k_values=[5, 10, 15, 20, 50]):
    """
    Avaliação CRAG Paper-Compliant:
    - Train Strict (Liked=1)
    - Eval Broad (Liked!=0)
    - OOV Penalty (Conta desconhecidos no denominador)
    """
    
    W = ease_model.B.copy()
    np.fill_diagonal(W, 0)
    n_items = W.shape[0]
    
    results = {k: {'recalls': [], 'ndcgs': []} for k in k_values}
    
    for conv in tqdm(dataset, desc="CRAG Eval (OOV Penalty)"):
        # ... (código de metadados init_q, rq, mentions igual) ...
        init_q = conv.get("initiatorQuestions", {})
        if isinstance(init_q, list): init_q = {}
        rq = conv.get("respondentQuestions", {})
        if isinstance(rq, list): rq = {}
        mentions = conv.get("movieMentions", {})
        if not mentions: continue
        mention_keys = list(mentions.keys())
        
        # ... (reconstrução timeline igual) ...
        timeline = []
        for msg in conv.get("messages", []):
            text = msg.get("text", "")
            ids_in_msg = [int(mid) for mid in mention_keys if f"@{mid}" in text]
            if ids_in_msg:
                sender = msg.get("senderWorkerId")
                timeline.append({
                    "ids": ids_in_msg,
                    "is_rec": (sender == conv.get("respondentWorkerId")),
                    "is_seek": (sender == conv.get("initiatorWorkerId"))
                })
        
        context_pool = set()
        seen_pool = set()
        
        for step in timeline:
            step_ids = step["ids"]
            
            if step["is_rec"]:
                targets = []
                
                for mid in step_ids:
                    mid_str = str(mid)
                    
                    # 1. Sugestão do Sistema (Obrigatório)
                    if rq.get(mid_str, {}).get("suggested") != 1: continue
                    
                    # 2. Novidade (Strict Discovery)
                    # Exclui apenas se o usuário DISSE que viu. (Aceita 0 e 2)
                    if init_q.get(mid_str, {}).get('seen') != 0: continue

                    # 3. Sentimento (Broad Evaluation)
                    # Aceita tudo que NÃO é Dislike explícito (0).
                    if init_q.get(mid_str, {}).get('liked') != 1: continue
                    
                    # 4. Histórico da Conversa
                    if mid in seen_pool: continue
                    
                    # 5. OOV (Out-of-Vocabulary) - MUDANÇA CRUCIAL
                    # NÃO fazemos "continue" se não estiver no movie_mapper.
                    # Nós ADICIONAMOS ao target para ele contar no denominador.
                    targets.append(mid)
                
                # Só roda inferência se houver contexto
                if targets and context_pool:
                    # Filtra apenas o que é possível mapear para o input do modelo
                    input_indices = [movie_mapper[mid] for mid in context_pool if mid in movie_mapper]
                    
                    if input_indices:
                        user_vector = np.zeros(n_items, dtype=np.float32)
                        user_vector[input_indices] = 1.0
                        scores = user_vector @ W
                        
                        seen_indices = [movie_mapper[mid] for mid in seen_pool if mid in movie_mapper]
                        if seen_indices:
                            scores[seen_indices] = -np.inf
                        
                        max_k = max(k_values)
                        top_indices = np.argpartition(scores, -max_k)[-max_k:]
                        top_scores = scores[top_indices]
                        sorted_idx = np.argsort(top_scores)[::-1]
                        top_k_sorted = top_indices[sorted_idx]
                        
                        # Mapeia targets para índices (APENAS OS CONHECIDOS)
                        # Os desconhecidos ficam de fora deste set, logo nunca darão "match" (Hit=0)
                        target_indices = {movie_mapper[t] for t in targets if t in movie_mapper}
                        
                        # O denominador inclui TUDO (Conhecidos + Desconhecidos)
                        num_gt = len(targets)
                        
                        for k in k_values:
                            hits = np.array([1 if idx in target_indices else 0 for idx in top_k_sorted[:k]])
                            
                            results[k]['recalls'].append(recall_at_k(hits, num_gt, k))
                            results[k]['ndcgs'].append(ndcg_at_k(hits, num_gt, k))
            
            seen_pool.update(step_ids)
            
            if step["is_seek"]:
                for mid in step_ids:
                    # Contexto continua Strict (Liked=1) para não poluir o input
                    if str(mid) in init_q and init_q[str(mid)].get('liked') != 1:
                        continue
                    context_pool.add(mid)
    
    # ... (prints finais iguais) ...
    # (Copie o final da função original)

     # 5. Relatório Final
    print("\n" + "="*80)
    print("📊 CRAG OFFLINE - EASE BASELINE (PAPER EXACT - MULTI K)")
    print("="*80)
    
    header = f"{'Metric':<12}" + "".join([f"K={k:<10}" for k in k_values])
    print(header)
    print("-" * 80)
    
    final_recalls = [np.mean(results[k]['recalls']) for k in k_values]
    final_ndcgs = [np.mean(results[k]['ndcgs']) for k in k_values]
    
    row_rec = f"{'Recall':<12}" + "".join([f"{val:<10.4f}" for val in final_recalls])
    row_ndcg = f"{'NDCG':<12}" + "".join([f"{val:<10.4f}" for val in final_ndcgs])
    
    print(row_rec)
    print(row_ndcg)
    print(f"\n✅ Total de Turnos Avaliados: {len(results[k_values[0]]['recalls'])}")
    print("="*80)
    
    print("\n📌 Benchmark (Aprox. Fig 6b do Paper):")
    print("   R@10 ≈ 0.11  |  R@20 ≈ 0.15")
    print("   N@10 ≈ 0.06  |  N@20 ≈ 0.08")
    
    return results


# NAO SEI, VER
def avaliar_llm(dataset, extracted_ids_list, ease_model, movie_mapper, k_values=[10, 20, 50]):
    """
    Avaliação Definitiva do Pipeline: LLM (Extração) -> EASE (Recomendação).
    
    LÓGICA PAPER-COMPLIANT:
    1. Input (Contexto): STRICT (Apenas liked=1 entra).
    2. Avaliação (Targets): BROAD (Aceita neutros/implícitos, rejeita apenas liked=0).
    3. Penalidade OOV: Itens desconhecidos contam como erro (entram no denominador).
    """
    
    # 1. Prepara Matriz EASE
    W = ease_model.B.copy()
    np.fill_diagonal(W, 0)
    n_items = W.shape[0]
    
    results = {k: {'recalls': [], 'ndcgs': []} for k in k_values}
    
    stats = {
        'turns_evaluated': 0,
        'skipped_no_context': 0,
        'skipped_no_targets': 0,
        'items_extracted_by_llm': 0,
        'targets_blocked_by_history': 0
    }
    
    print(f"🔬 Iniciando Avaliação do Pipeline LLM ({len(dataset)} conversas)...")
    
    # Zipa dataset com as respostas da LLM
    for conv, llm_recognized_ids in tqdm(zip(dataset, extracted_ids_list), total=len(dataset)):
        
        mentions = conv.get("movieMentions", {})
        if not mentions: continue
        
        context_pool = set()
        seen_pool = set()
        
        seeker_id = conv.get("initiatorWorkerId")
        recommender_id = conv.get("respondentWorkerId")
        
        # Sanitização
        init_q = conv.get("initiatorQuestions", {})
        if isinstance(init_q, list): init_q = {}
        rq = conv.get("respondentQuestions", {})
        if isinstance(rq, list): rq = {}
        
        mention_keys = list(mentions.keys())
        
        for msg in conv.get("messages", []):
            text = msg.get("text", "")
            sender = msg.get("senderWorkerId")
            
            # Acha IDs mencionados nesta mensagem
            ids_found = re.findall(r'@(\d+)', text)
            ids_in_msg = [int(mid) for mid in ids_found if mid in mention_keys]
            
            if not ids_in_msg: continue
            
            # --- TURNO DO USUÁRIO (Contexto - STRICT) ---
            if sender == seeker_id:
                for mid in ids_in_msg:
                    mid_str = str(mid)
                    
                    # 1. A LLM reconheceu? (Filtro da Extração)
                    if mid_str not in llm_recognized_ids:
                        continue
                        
                    # 2. CONDIZÊNCIA: É positivo explícito? (Strict Context)
                    # Para INPUT, só usamos o que é certeza (liked=1)
                    val = init_q.get(mid_str, {}).get('liked')
                    if val != 1: 
                        continue 
                        
                    context_pool.add(mid)
                    stats['items_extracted_by_llm'] += 1
                    seen_pool.add(mid)

            # --- TURNO DO SISTEMA (Targets - BROAD + OOV Penalty) ---
            elif sender == recommender_id:
                targets = []
                for mid in ids_in_msg:
                    mid_str = str(mid)
                    
                    # 1. Sugestão do Sistema
                    if rq.get(mid_str, {}).get('suggested') != 1: continue
                    
                    # 2. Novidade (Remove apenas se DISSE que viu)
                    if init_q.get(mid_str, {}).get('seen') != 0: continue
                    
                    # 3. Sentimento (Remove apenas se DISSE que não gostou)
                    # Aceita 1 (Like), 2 (Neutro/Aceitou) e None
                    if init_q.get(mid_str, {}).get('liked') != 1: continue 
                    
                    # 4. Histórico
                    if mid in seen_pool: 
                        stats['targets_blocked_by_history'] += 1
                        continue
                    
                    # 5. OOV Penalty: ADICIONAMOS mesmo se não estiver no mapper
                    # Isso garante que o denominador (num_gt) seja o "Real"
                    targets.append(mid)
                
                # Validações para inferência
                if not context_pool:
                    stats['skipped_no_context'] += 1
                    seen_pool.update(ids_in_msg)
                    continue
                    
                if not targets:
                    stats['skipped_no_targets'] += 1
                    seen_pool.update(ids_in_msg)
                    continue

                # --- INFERÊNCIA EASE ---
                input_indices = [movie_mapper[mid] for mid in context_pool if mid in movie_mapper]
                
                if not input_indices:
                    stats['skipped_no_context'] += 1
                    seen_pool.update(ids_in_msg)
                    continue

                user_vector = np.zeros(n_items, dtype=np.float32)
                user_vector[input_indices] = 1.0
                scores = user_vector @ W
                
                # Masking
                mask_ids = seen_pool | context_pool
                mask_indices = [movie_mapper[mid] for mid in mask_ids if mid in movie_mapper]
                if mask_indices:
                    scores[mask_indices] = -np.inf
                
                # Métricas
                max_k = max(k_values)
                top_indices = np.argpartition(scores, -max_k)[-max_k:]
                top_scores = scores[top_indices]
                sorted_idx = np.argsort(top_scores)[::-1]
                top_k_sorted = top_indices[sorted_idx]
                
                # Mapeia targets para índices (APENAS OS CONHECIDOS)
                # OOV nunca dará match (Hit=0), mas contou no len(targets)
                target_indices = {movie_mapper[t] for t in targets if t in movie_mapper}
                num_gt = len(targets)
                
                for k in k_values:
                    hits = np.array([1 if idx in target_indices else 0 for idx in top_k_sorted[:k]])
                    results[k]['recalls'].append(recall_at_k(hits, num_gt, k))
                    results[k]['ndcgs'].append(ndcg_at_k(hits, num_gt, k))
                
                stats['turns_evaluated'] += 1
                seen_pool.update(ids_in_msg)

    # ============================================
    # RELATÓRIO FINAL
    # ============================================
    print("\n" + "="*80)
    print("🚀 RESULTADOS FINAIS: Pipeline LLM + EASE (PAPER COMPLIANT)")
    print("="*80)
    
    header = f"{'Metric':<12}" + "".join([f"K={k:<10}" for k in k_values])
    print(header)
    print("-" * 80)
    
    final_recalls = [np.mean(results[k]['recalls']) if results[k]['recalls'] else 0.0 for k in k_values]
    final_ndcgs = [np.mean(results[k]['ndcgs']) if results[k]['ndcgs'] else 0.0 for k in k_values]
    
    print(f"{'Recall':<12}" + "".join([f"{val:<10.4f}" for val in final_recalls]))
    print(f"{'NDCG':<12}" + "".join([f"{val:<10.4f}" for val in final_ndcgs]))
    
    print("\n📊 DIAGNÓSTICO:")
    print(f"Turnos Avaliados: {stats['turns_evaluated']}")
    print(f"Total Itens Contexto (LLM): {stats['items_extracted_by_llm']}")
    print(f"Skipped (Sem Contexto Válido): {stats['skipped_no_context']}")
    print("="*80)
    
    return results, stats



