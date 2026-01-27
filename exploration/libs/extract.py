import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from editdistance import eval as edit_distance
import re

# CERTO, NAO MUDAR
def clean_data(dataset):
    """
    Remove diálogos inválidas do dataset 
    """

    invalid_indexes = []

    # Identifica diálogos inválidas
    for i, conv in enumerate(dataset):
        if (
            isinstance(conv.get("movieMentions"), list) # Se for lista em vez de dicionário
            or not isinstance(conv.get("movieMentions", {}), dict) # Se dicionário for inválido
            or "messages" not in conv # Diálogo sem mensagem
            or not conv["messages"] # Diálogo sem mensagem
        ):
            invalid_indexes.append(i)

    # Remove invalidos
    for i in sorted(invalid_indexes, reverse=True):
        del dataset[i]

    print(f"✅ Conversas válidas restantes: {len(dataset)}")
    print(f"🗑️ Conversas removidas: {len(invalid_indexes)}")


# def extract_interactions(conv):
#     """
#     1. Usa movieMentions (Robustez de extração).
#     2. MAS filtra explícitos 'liked=0' (Protocolo CRAG: remove negativos).
#     3. Define User = ConversationId (Protocolo CRAG: Pseudo-user).
#     """
#     rows = []
    
#     # OPÇÃO A: Protocolo CRAG estrito (Pseudo-User) 
#     # user_id = conv.get("conversationId") 
    
#     # OPÇÃO B: Sua abordagem (User real - Geralmente melhor performance)
#     user_id = conv.get("conversationId")
    
#     # O "ouro" da extração textual
#     mentions = conv.get("movieMentions", {})
    
#     # O "ouro" do sentimento (para filtrar negativos)
#     # Precisamos checar tanto o initiator quanto o respondent para saber o sentimento
#     init_q = conv.get("initiatorQuestions", {})
#     resp_q = conv.get("respondentQuestions", {})
    
#     # Helper para checar sentimento
#     def is_explicitly_liked(mid_str):
#         # Verifica Iniciador (Seeker)
#         if mid_str in init_q:
#             val = init_q[mid_str].get('liked')
#             # Aceita se for 1 OU 2 (ou True)
#             if val == 1:
#                 return True
#         return False

#     if user_id and isinstance(mentions, dict):
#         for movie_id_str in mentions.keys():
#             try:
#                 # CRAG: "exclude these items [negatively mentioned]" 
#                 if is_explicitly_liked(movie_id_str):
#                     rows.append((user_id, int(movie_id_str), 1))
                
#             except ValueError:
#                 continue
                
#     return rows



# def build_interaction_df(dataset):
#     all_rows = []
#     for conv in dataset:
#         all_rows.extend(extract_interactions(conv))
        
#     df = pd.DataFrame(all_rows, columns=["userId", "movieId", "rating"])
#     # Remove duplicatas (se o filme foi mencionado 2x na mesma conversa, conta como 1)
#     df = df.drop_duplicates()
#     return df

# NAO SEI, VER
def extract_interactions(conv, conv_idx):
    """
    1. Usa movieMentions (Robustez de extração).
    2. Define User = Índice da Conversa (Session-based puro: 0, 1, 2...).
    3. Input Permissivo: Aceita 1 (Like) e 2 (Mention/Did not say) para densidade.
    """
    rows = []
    
    # MUDANÇA AQUI: O ID do usuário passa a ser o índice sequencial da lista
    # Isso garante unicidade absoluta por sessão (Session-based)
    user_id = conv_idx
    
    # O "ouro" da extração textual
    mentions = conv.get("movieMentions", {})
    
    # Metadados para filtrar dislikes
    init_q = conv.get("initiatorQuestions", {})


    # Helper para checar sentimento (Ajustado para Input Permissivo)
    def is_valid_input(mid_str):
        # Se não tem metadados (init_q), assumimos que a menção é válida (Neutro/2 implícito)
        if mid_str not in init_q:
            return True 
            
        val = init_q[mid_str].get('liked')
        
        # Se for 1 (like), retorna True
        if val == 1:
            return True
            
        return False

    if isinstance(mentions, dict):
        for movie_id_str in mentions.keys():
            try:
                # CRAG: "exclude these items [negatively mentioned]" 
                if is_valid_input(movie_id_str):
                    rows.append((user_id, int(movie_id_str), 1))
                
            except ValueError:
                continue
                
    return rows

# NAO SEI, VER
def build_interaction_df(dataset):
    all_rows = []
    
    # MUDANÇA AQUI: Usamos enumerate para gerar o ID sequencial (0, 1, 2...)
    # Exatamente como "for instance in instances", mas capturando o índice.
    
    for idx, conv in tqdm(enumerate(dataset), total=len(dataset)):
        # Passamos 'conv' e o 'idx' para a função
        all_rows.extend(extract_interactions(conv, idx))
        
    df = pd.DataFrame(all_rows, columns=["userId", "movieId", "rating"])
    
    # Remove duplicatas (se o filme foi mencionado 2x na mesma conversa)
    df = df.drop_duplicates()
    
    return df   


# CERTO, NAO MUDAR
def create_sparse_matrix(df, user_mapper, movie_mapper, shape):
    """
        Cria matriz esparsa a partir de DataFrame
        user_mapper e movie_mapper são dicionarios do mapeamento
    """

    df_mapped = df.copy()

    # Faz mapeamento pois matrizes esparsas só aceitam índices inteiros começando em 0, e nao podem ser enormes
    
    df_mapped['user_index'] = df_mapped['userId'].map(user_mapper)#.astype('Int64')
    df_mapped['movie_index'] = df_mapped['movieId'].map(movie_mapper)#.astype('Int64')
    
    # remove linhas com NaN em índices ou rating inválido
    df_mapped = df_mapped.dropna(subset=['user_index', 'movie_index', 'rating'])
    df_mapped = df_mapped[df_mapped['rating'] > 0]

    # Remove duplicatas
    df_mapped = df_mapped.drop_duplicates(subset=['user_index', 'movie_index'], keep='last')
    
    # Cria matriz esparsa
    matrix = csr_matrix(
        (df_mapped['rating'].astype(float),
         (df_mapped['user_index'], #.astype('Int64')
          df_mapped['movie_index'])),#.astype('Int64')
        shape=shape
    )
    return matrix



# CERTO, NAO MUDAR
def safe_get_flags(conv, key):
    # Acessa conversas nulas 
    data = conv.get(key, {})
    if isinstance(data, list): 
        return {}
    return data

# NAO SEI, VER
def get_ground_truth(conv):

    true_ids = set()
    
    # Busca conversas não nulas
    init_q = safe_get_flags(conv, 'initiatorQuestions')
    #resp_q = safe_get_flags(conv, 'respondentQuestions')

    
    # User
    for mid, flags in init_q.items():
        if flags.get('liked') == 1:
            true_ids.add(str(mid))
    
    # # System
    # for mid, flags in resp_q.items():
    #     if flags.get('liked') in POSITIVE_VALUES:
    #         true_ids.add(str(mid))
            
    return true_ids
  


# NAO SEI, VER
def fuzzy_match(llm_text, mentions):
    """
    Recebe o texto (já normalizado/limpo) e o dicionário de menções daquela conversa.
    Retorna: Um set de IDs (strings) prontos para o sistema.
    
    Trata casos especiais:
    - Remove anos entre parênteses: "American Pie (1999)" → "american pie"
    - Remove artigos opcionalmente para fuzzy matching
    """
    ids_encontrados = set()
    
    if not llm_text: 
        return ids_encontrados

    def clean(t):
        """
        Normaliza texto:
        1. Remove ano (1999), (2010), etc
        2. Remove pontuação
        3. Converte para minúsculas
        4. Remove espaços extras
        """
        # Remove ano entre parênteses (formato: (YYYY))
        text = re.sub(r'\s*\(\d{4}\)', '', t)
        
        # Remove pontuação
        text = re.sub(r'[^\w\s]', '', text)
        
        # Minúsculas e trim
        text = text.strip().lower()
        
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def clean_without_articles(t):
        """
        Versão alternativa que também remove artigos comuns
        Útil para fuzzy matching mais robusto
        """
        text = clean(t)
        
        # Remove artigos do início
        for article in ['the ', 'a ', 'an ']:
            if text.startswith(article):
                text = text[len(article):]
                break
        
        return text
    
    # Prepara dicionário reverso {nome_limpo: id}
    title_to_id = {clean(v): k for k, v in mentions.items()}
    candidates = list(title_to_id.keys())
    
    # Dicionário alternativo sem artigos (para fallback)
    title_to_id_no_articles = {clean_without_articles(v): k for k, v in mentions.items()}
    candidates_no_articles = list(title_to_id_no_articles.keys())

    for line in llm_text.split('\n'):
        if '####' not in line: 
            continue
        
        raw_name = line.split('####')[0].strip()
        name_clean = clean(raw_name)
        
        if not name_clean: 
            continue
        
        found_id = None
        
        # 1. Match Exato
        if name_clean in title_to_id:
            found_id = title_to_id[name_clean]
        
        # 2. Match Parcial (substring)
        elif any(name_clean in db_name for db_name in candidates):
            for db_name, db_id in title_to_id.items():
                if name_clean in db_name:
                    found_id = db_id
                    break
        
        # 3. Fuzzy Match com Edit Distance
        elif candidates:
            min_distance = float('inf')
            best_match = None
            
            for candidate in candidates:
                dist = edit_distance(name_clean, candidate)
                max_len = max(len(name_clean), len(candidate))
                normalized_dist = dist / max_len if max_len > 0 else 1.0
                
                # Threshold: aceita se similaridade >= 70% (dist <= 30%)
                if normalized_dist <= 0.25 and dist < min_distance:
                    min_distance = dist
                    best_match = candidate
            
            if best_match:
                found_id = title_to_id[best_match]
        
        # 4. Fallback: Tenta sem artigos
        if not found_id:
            name_no_articles = clean_without_articles(raw_name)
            
            # Match exato sem artigos
            if name_no_articles in title_to_id_no_articles:
                found_id = title_to_id_no_articles[name_no_articles]
            
            # Fuzzy sem artigos
            elif candidates_no_articles:
                min_distance = float('inf')
                best_match = None
                
                for candidate in candidates_no_articles:
                    dist = edit_distance(name_no_articles, candidate)
                    max_len = max(len(name_no_articles), len(candidate))
                    normalized_dist = dist / max_len if max_len > 0 else 1.0
                    
                    if normalized_dist <= 0.30 and dist < min_distance:
                        min_distance = dist
                        best_match = candidate
                
                if best_match:
                    found_id = title_to_id_no_articles[best_match]
        
        if found_id:
            ids_encontrados.add(str(found_id))
            
    return ids_encontrados

# NAO SEI, VER
def ids_dataset_fuzzy(dataset, responses_list):
    """
    Roda o Fuzzy Match em todo o dataset UMA ÚNICA VEZ.
    Retorna uma lista de sets, onde cada posição corresponde a uma conversa.
    """
    print("⚙️ Processando Extração de Entidades (Fuzzy Match)...")
    all_extracted_ids = []
    
    # Zipamos o dado real (que tem as mentions) com a resposta da LLM
    for conv, llm_text in zip(dataset, responses_list):
        mentions = conv.get("movieMentions", {})
        
        # Chama a função pura do Passo 1
        ids = fuzzy_match(llm_text, mentions)
        
        all_extracted_ids.append(ids)
        
    print(f"✅ Processamento concluído! {len(all_extracted_ids)} conversas mapeadas.")
    return all_extracted_ids



# NAO SEI, VER
def create_binary_vectors(true_set, pred_set, universe_ids):
    """
    Transforma os sets em vetores binários alinhados com o universo de filmes da conversa.
    Necessário para usar o jaccard_score do sklearn.
    """
    # Cria vetores de 0s e 1s para comparação
    y_true = [1 if mid in true_set else 0 for mid in universe_ids]
    y_pred = [1 if mid in pred_set else 0 for mid in universe_ids]
    
    return y_true, y_pred

# CERTO, NAO MUDAR
def conv_to_text(dataset):
    # Lista que vai guardar os textos de todas as conversas
    all_formatted_texts = []

    for conv in dataset:
        
        user_id = conv['initiatorWorkerId']      # Usuário 
        recommender_id = conv['respondentWorkerId'] # Sistema 

        formatted_text_lines = []

        # Loop pelas mensagens da conversa atual
        for msg in conv.get('messages', []):
            sender_id = msg['senderWorkerId']
            text = msg['text'] 
            
            # Identifica de quem é a msg
            role = "Unknown" # Valor padrão
            if sender_id == user_id:
                role = "User"
            elif sender_id == recommender_id:
                role = "System"

            # Formata a linha e adiciona
            formatted_text_lines.append(f"{role}: {text}")
        
        # Junta as linhas dessa conversa e adiciona na lista final
        full_conversation = "\n".join(formatted_text_lines)
        all_formatted_texts.append(full_conversation)
    
    return all_formatted_texts


# CERTO, NAO MUDAR
# Remover -2 -1 e 0, trocar 2 por 1
def normalizar_scores_llm(lista_respostas):
    """
    Processa a lista de saídas da LLM aplicando as regras:
    1. Remove linhas com score -1 ou -2 (Negativos).
    2. Transforma scores 0, 1 e 2 em 1 (Positivos unificados).
    
    Retorna:
    --------
    list: Lista com o mesmo tamanho, mas com strings filtradas e normalizadas.
    """
    lista_processada = []

    for resposta in lista_respostas:
        linhas_novas = []
        
        # Divide a string em linhas para processar cada filme
        for linha in resposta.split('\n'):
            # Verificação básica de formato
            if '####' in linha:
                partes = linha.split('####')
                
                # Garante que temos título e score
                if len(partes) >= 2:
                    titulo = partes[0].strip()
                    try:
                        score_original = int(partes[1].strip())
                        
                        # REGRA 1: Remover negativos
                        if score_original < 0:
                            continue
                        
                        # REGRA 2: Transformar 0, 1 e 2 em 1
                        # Se passou pelo filtro acima, é >= 0, então vira 1.
                        linhas_novas.append(f"{titulo}####1")
                        
                    except ValueError:
                        continue # Pula se o score não for número
        
        # Reconstrói a string com quebras de linha
        lista_processada.append('\n'.join(linhas_novas))
        
    return lista_processada

# CERTO, NAO MUDAR
def responses_list_to_df(responses_list):
    rows = []

    for conv_id, response in enumerate(responses_list):
        if not isinstance(response, str):
            continue
        
        # cada filme está em uma linha
        lines = response.strip().split("\n")
        
        for line in lines:
            try:
                title, attitude = line.split("####")
                rows.append({
                    "conversationId": conv_id, 
                    "movieTitle": title.strip(),
                    "rating": int(attitude.strip())
                })
            except ValueError:
                # ignora linhas mal formatadas
                continue

    return pd.DataFrame(rows)
