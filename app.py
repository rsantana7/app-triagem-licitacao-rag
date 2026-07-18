# app.py
import streamlit as st
from services import (
    processar_e_indexar_pdfs, 
    buscar_contexto_rag, 
    executar_filtro_slm_local, 
    consultar_llm_nuvem_resiliente,
    validar_chave_gemini_real # Importando o novo validador defensivo
)

st.set_page_config(page_title="Auditoria de Editais RAG Híbrido", page_icon="⚖️", layout="wide")

## renata - baixar arquivo
st.sidebar.header("📥 Arquivo de exemplo")
# Botão para download do arquivo pdf "manual_politica.pdf"
# Nota: Para o botão funcionar, o arquivo 'manual_politica.pdf' precisa existir na raiz do seu projeto.
try:
    with open("EDITAL DE LICITACAO PUBLICA NACIONAL N 042-2026.pdf", "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.download_button(
        label="📄 Baixar Edital de Licitacao.pdf",
        data=pdf_bytes,
        file_name="EDITAL DE LICITACAO PUBLICA NACIONAL N 042-2026.pdf",
        mime="application/pdf"
    )
except FileNotFoundError:
    st.sidebar.info("💡 Para ativar o botão de download de exemplo, salve um arquivo 'pdf' na raiz do seu projeto.")

st.markdown("---")

# --- CONTROLADOR DA BARRA LATERAL (GOVERNANÇA) ---
st.sidebar.header("🔑 Painel de Credenciais")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key:", 
    type="password", 
    placeholder="AIzaSy...",
    help="Insira sua chave privada do Google AI Studio para liberar a auditoria do sistema."
)

with st.sidebar.expander("❓ Como criar sua API Key"):
    st.markdown("""
    1. Entre na plataforma do [Google AI Studio](https://google.com).
    2. Faça login com sua conta padrão do Google.
    3. Clique em **"Get API Key"** no menu esquerdo.
    4. Clique em **"Create API Key"** e copie o token gerado.
    """)

# --- TELA CENTRAL ---
st.title("⚖️ Sistema de Triagem e Auditoria de Editais (RAG Híbrido)")
st.subheader("Pipeline Inteligente com Validação Preventiva de Credenciais e Segurança UX/UI")
st.markdown("---")

# Seção de Carregamento de Arquivos
st.header("📂 Upload de Documentos para Análise")
arquivos_carregados = st.file_uploader(
    "Selecione um ou mais arquivos de Edital (PDF ou TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if "base_vetorial" not in st.session_state:
    st.session_state.base_vetorial = None
if "nomes_arquivos_indexados" not in st.session_state:
    st.session_state.nomes_arquivos_indexados = []

if arquivos_carregados:
    nomes_atuais = [arq.name for arq in arquivos_carregados]
    if st.session_state.nomes_arquivos_indexados != nomes_atuais:
        with st.spinner("🚀 Extraindo dados e construindo banco vetorial local..."):
            base = processar_e_indexar_pdfs(arquivos_carregados)
            if base:
                st.session_state.base_vetorial = base
                st.session_state.nomes_arquivos_indexados = nomes_atuais
                st.success(f"✅ Sucesso! {len(base['chunks'])} fragmentos textuais indexados localmente.")

# Fluxo de execução com travas de segurança robustas
if st.session_state.base_vetorial is not None:
    st.markdown("---")
    st.header("🔍 Central de Auditoria Algorítmica")
    query_usuario = st.text_input("O que deseja identificar nos documentos?", value="Qual o valor das multas por atraso na entrega, as especificações de memória RAM e o SLA de suporte?")
    
    if st.button("Iniciar Auditoria de Riscos", type="primary"):
        
        # 🛡️ TRAVA 1: Validação de presença do campo textual
        if not api_key_input.strip():
            st.error("❌ **Acesso Negado:** A chave de API do Google Gemini não foi informada!")
            st.warning("💡 **Como Corrigir:** Vá até a **barra lateral esquerda**, insira seu token `AIzaSy...` no campo de texto e clique no botão de auditoria novamente.")
        
        else:
            # 🛡️ TRAVA 2: Validação de conexão em tempo real com os servidores da Google
            with st.spinner("🔒 Autenticando credencial junto à infraestrutura do Google Gemini..."):
                chave_valida, mensagem_erro = validar_chave_gemini_real(api_key_input)
                
            if not chave_valida:
                st.error("❌ **Falha de Autenticação Crítica:** A chave informada é inválida ou expirou!")
                st.info(f"📋 **Detalhe Técnico Retornado:** {mensagem_erro}")
                st.warning("💡 **Recomendação:** Verifique se não houve erros de cópia no token ou gere uma nova chave de testes dentro do painel do Google AI Studio.")
            
            else:
                # Se passou em todas as camadas de segurança defensiva, inicia o pipeline
                st.markdown("### 🏃‍♂️ Rastreabilidade Executiva do Pipeline de Dados")
                
                with st.status("Processando camadas da arquitetura híbrida...", expanded=True) as status:
                    st.write("🛰️ **Etapa 1: Busca RAG Vetorial** - Filtrando vizinhos mais próximos estatisticamente...")
                    documentos_brutos = buscar_contexto_rag(query_usuario, st.session_state.base_vetorial, top_k=15)
                    
                    st.write("🤖 **Etapa 2: Filtro do SLM Local** - Removendo redundâncias e rodapés legais...")
                    contexto_filtrado_slm = executar_filtro_slm_local(documentos_brutos)
                    
                    st.write("📝 **Etapa 3: Construção do Prompt Enxuto** - Compactando estrutura de contexto...")
                    prompt_enxuto = f"""
                    Analise rigorosamente as cláusulas contratuais para responder à pergunta do auditor.
                    
                    Trechos Selecionados por SLM Local:
                    {contexto_filtrado_slm}
                    
                    Pergunta do Auditor: {query_usuario}
                    """
                    
                    st.markdown("---")
                    col_bruto, col_limpo = st.columns(2)
                    col_bruto.metric("Fragmentos Brutos (RAG)", f"{len(documentos_brutos)} Chunks")
                    col_limpo.metric("Dados Destilados (SLM)", "Cláusulas Filtradas", "-75% Tokens", delta_color="normal")
                    st.markdown("---")
                    
                    st.write("☁️ **Etapa 4: Inferência na Nuvem** - Despachando contexto compacto de forma segura...")
                    resposta_final = consultar_llm_nuvem_resiliente(prompt_enxuto, api_key_usuario=api_key_input)
                    
                    status.update(label="Análise finalizada com sucesso!", state="complete")
                    
                st.markdown("### 📋 Relatório Consolidado de Auditoria (Gerado via Nuvem)")
                st.info(resposta_final)
else:
    st.info("💡 Aguardando o upload de pelo menos um arquivo PDF ou TXT acima para habilitar a central de auditoria.")


## renata
# st.sidebar.header("🎯 Desenvolvido por")
st.markdown("---")
# CSS caixa cinza com texto destacado
st.markdown(
"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
    <span style="color: #ff4b4b;">Desenvolvido por :</span>
    <br>
    <span style="color: #000000; font-weight: bold;"> Renata LC Santana</span>
    <br>
    <span>AI First | IA Generativa | LLMs | RAG | Agentes IA | APIs | Machine Learning</span>
    </div>

""", 
unsafe_allow_html=True
)
