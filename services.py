# services.py
import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import APIError

def obter_cliente_gemini(api_key_usuario=None):
    """Instancia o cliente oficial priorizando o input do usuário na interface."""
    api_key = api_key_usuario or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        raise ValueError("Chave de API do Google Gemini não encontrada.")
    return genai.Client(api_key=api_key)

def validar_chave_gemini_real(api_key_usuario=None):
    """
    Executa uma validação real e rápida na infraestrutura do Google Gemini.
    Retorna (True, None) se estiver ok, ou (False, mensagem_erro) se falhar.
    """
    try:
        client = obter_cliente_gemini(api_key_usuario)
        # Executa uma listagem de modelos simplificada para checar as credenciais na nuvem
        client.models.list(config={'page_size': 1})
        return True, None
    except APIError as e:
        return False, f"A chave fornecida foi recusada pelos servidores do Google (Erro da API: {e.message})."
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Falha inesperada ao tentar validar a credencial: {str(e)}"

def processar_e_indexar_pdfs(arquivos_carregados):
    """Extrai o texto real de múltiplos PDFs carregados e quebra em fragmentos."""
    fragmentos_texto = []
    for arquivo in arquivos_carregados:
        try:
            if arquivo.name.endswith('.txt'):
                texto_completo = arquivo.read().decode("utf-8")
                paragrafos = [p.strip() for p in texto_completo.split('\n\n') if p.strip()]
                for p in paragrafos:
                    fragmentos_texto.append(f"[{arquivo.name}] {p}")
            else:
                leitor_pdf = PdfReader(arquivo)
                for num_pag, pagina in enumerate(leitor_pdf.pages):
                    texto_pag = pagina.extract_text()
                    if texto_pag and texto_pag.strip():
                        linhas = texto_pag.split('\n\n')
                        for linha in linhas:
                            if len(linha.strip()) > 30:
                                fragmentos_texto.append(f"[{arquivo.name} - Pág {num_pag+1}] {linha.strip()}")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo {arquivo.name}: {str(e)}")
            
    if not fragmentos_texto:
        return None
        
    vectorizer = TfidfVectorizer(stop_words=None)
    matriz_embeddings = vectorizer.fit_transform(fragmentos_texto)
    
    return {
        "chunks": fragmentos_texto,
        "vectorizer": vectorizer,
        "embeddings": matriz_embeddings
    }

def buscar_contexto_rag(query, base_vetorial, top_k=15):
    """Fase 1: Executa a busca por similaridade de cosseno na base de dados indexada."""
    vectorizer = base_vetorial["vectorizer"]
    matriz_embeddings = base_vetorial["embeddings"]
    chunks = base_vetorial["chunks"]
    
    vetor_query = vectorizer.transform([query])
    similaridades = cosine_similarity(vetor_query, matriz_embeddings).flatten()
    
    top_indices = np.argsort(similaridades)[::-1][:top_k]
    return [chunks[idx] for idx in top_indices if similaridades[idx] > 0.05]

def executar_filtro_slm_local(documentos_recuperados):
    """Fase 2: Filtro local eliminando redundâncias semânticas e ruídos."""
    linhas_vistas = set()
    contexto_limpo = []
    palavras_ruido = ["protocolo interno", "diário oficial", "assinatura eletrônica", "controle de versão"]
    
    for doc in documentos_recuperados:
        if any(termo in doc.lower() for termo in palavras_ruido):
            continue
        assinatura = "".join([c for c in doc.lower() if c.isalnum()])[:30]
        if assinatura not in linhas_vistas:
            linhas_vistas.add(assinatura)
            contexto_limpo.append(doc)
    return "\n\n".join(contexto_limpo)

def consultar_llm_nuvem_resiliente(prompt_enxuto, api_key_usuario=None):
    """Fases 3 e 4: Envio tático ao cluster do Gemini com redundância automática contra erros 429."""
    try:
        client = obter_cliente_gemini(api_key_usuario)
    except ValueError as e:
        return f"⚠️ Erro de Autenticação: {str(e)}"

    sys_instruction = "Você é um Auditor Jurídico sênior especialista em Editais e Licitações Governamentais. Seja direto e cite as fontes."
    modelo_principal = "gemini-3.5-flash"
    modelo_fallback = "gemini-2.5-pro"

    try:
        response = client.models.generate_content(
            model=modelo_principal,
            contents=prompt_enxuto,
            config={"system_instruction": sys_instruction, "temperature": 0.1}
        )
        return response.text
    except APIError as e:
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            st.warning("⚠️ Limite de requisições esgotado (Erro 429). Ativando o plano de contingência para o Gemini Pro...")
            try:
                response = client.models.generate_content(
                    model=modelo_fallback,
                    contents=prompt_enxuto,
                    config={"system_instruction": sys_instruction, "temperature": 0.1}
                )
                return response.text
            except APIError as fb_err:
                return f"❌ Erro Crítico de Comunicação na API (Fallback): {fb_err.message}"
        else:
            return f"❌ Erro na API do Google Gemini: {e.message}"

