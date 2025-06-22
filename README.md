# Processamento de Imagens com OpenCV e Streamlit

Este projeto é um aplicativo web para processamento de imagens em Python, utilizando as bibliotecas OpenCV, NumPy, Matplotlib e Streamlit. Permite aplicar filtros, transformações de intensidade, segmentação e operações morfológicas em imagens.

## Funcionalidades
- Upload de imagens (PNG, JPG, JPEG, BMP, TIF)
- Filtros passa-baixa: média, mediana, gaussiano, máximo, mínimo
- Filtros passa-alta: Laplaciano, Roberts, Prewitt, Sobel
- Transformações de intensidade: alargamento de contraste, equalização de histograma
- Segmentação: Otsu
- Morfologia matemática: erosão e dilatação
- Download das imagens processadas

## Instalação

1. **Clone este repositório:**
```bash
git clone https://github.com/estefanialopes/Projeto_PDI.git
cd Projeto_PDI
```

2. **Crie um ambiente virtual (opcional, mas recomendado):**
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

## Execução

Execute o aplicativo Streamlit:
```bash
streamlit run app.py
```

Acesse o endereço mostrado no terminal (geralmente http://localhost:8501).

## Estrutura dos arquivos
- `app.py`: Interface principal do Streamlit
- `filtros.py`: Funções de processamento de imagem
- `requirements.txt`: Lista de dependências

## Requisitos
- Python 3.8+

## Observações
- O app foi testado no Windows, mas deve funcionar em outros sistemas operacionais com Python instalado.
- Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Documentação detalhada está nos comentários do código-fonte.**
