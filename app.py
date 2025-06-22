import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
import filtros
import os
from descritores import color_descriptor, texture_descriptor, shape_descriptor


st.title("Processamento de Imagens com OpenCV e Matplotlib")

# CSS para todos os botões (ação e download): fundo branco, texto preto, borda azul, letras azuis ao passar mouse
st.markdown('''
    <style>
    .stButton > button, .stDownloadButton > button {
        background-color: #fff !important;
        color: #222 !important;
        border: 2px solid #2563eb !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        font-size: 1.08em !important;
        padding: 0.5em 1.8em !important;
        margin-top: 0.5em !important;
        margin-bottom: 0.5em !important;
        transition: background 0.2s, color 0.2s;
        display: flex;
        align-items: center;
        gap: 0.5em;
        box-shadow: none !important;
    }
    .stButton > button:hover, .stButton > button:focus, .stButton > button:active,
    .stDownloadButton > button:hover, .stDownloadButton > button:focus, .stDownloadButton > button:active {
        background-color: #fff !important;
        color: #2563eb !important;
        border: 2px solid #2563eb !important;
    }
    </style>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg", "bmp", "tif"])

if uploaded_file is not None:
    if 'results' not in st.session_state:
        st.session_state['results'] = []
    file_bytes = uploaded_file.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 0)
    st.header("Visualização Inicial")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(image, caption="Imagem Original", use_container_width=True, channels="GRAY")
        success, buffer = cv2.imencode('.png', image)
        if success:
            st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='imagem_original.png', mime='image/png')
    with col_img2:
        if st.button("Calcular Histograma da Imagem Original"):
            fig, ax = plt.subplots()
            ax.hist(img.ravel(), bins=256, range=[0,256])
            ax.set_title("Histograma da Imagem Original")
            st.pyplot(fig)

    
    
    
    
    
    

    st.header("Transformações de Intensidade")
    # Normalização (Alargamento de Contraste)
    if st.button("Alargamento de Contraste (normalize)"):
        stretch = filtros.apply_contrast_stretch(img)
        col_norm1, col_norm2 = st.columns(2)
        with col_norm1:
            st.image(stretch, caption="Contraste Alargado", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', stretch)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='alargamento_contraste.png', mime='image/png')
        with col_norm2:
            fig2, ax2 = plt.subplots()
            ax2.hist(stretch.ravel(), bins=256, range=[0,256])
            ax2.set_title("Histograma da Imagem Alargada")
            st.pyplot(fig2)

    # Equalização de Histograma
    if st.button("Equalização de Histograma"):
        eq = filtros.apply_histogram_equalization(img)
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.image(eq, caption="Equalização de Histograma", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', eq)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='equalizacao_histograma.png', mime='image/png')
        with col_eq2:
            fig3, ax3 = plt.subplots()
            ax3.hist(eq.ravel(), bins=256, range=[0,256])
            ax3.set_title("Histograma da Imagem Equalizada")
            st.pyplot(fig3)

    st.header("Filtros Passa-Baixa")
    min_dim = min(img.shape[0], img.shape[1])
    st.markdown("<b>Escolha o tamanho da máscara a ser aplicada (valor inteiro, ex: 3 para 3x3):</b>", unsafe_allow_html=True)
    ksize_pb = st.number_input("Tamanho da máscara (kernel)", min_value=1, max_value=min(99, min_dim), value=3, step=2, format="%d")
    if ksize_pb % 2 == 0:
        st.error("O tamanho da máscara deve ser um número ímpar.")
    elif ksize_pb > min_dim:
        st.error(f"O tamanho da máscara ({ksize_pb}) não pode ser maior que a menor dimensão da imagem ({min_dim}).")
    if st.button("Filtro Média"):
        if ksize_pb > img.shape[0] or ksize_pb > img.shape[1]:
            st.error(f"O kernel ({ksize_pb}x{ksize_pb}) é maior que a imagem. Escolha um valor menor.")
        elif ksize_pb <= 0:
            st.error("O kernel deve ser maior que zero.")
        else:
            media = filtros.apply_mean_filter(img, ksize=ksize_pb)
            col_media1, col_media2 = st.columns(2)
            with col_media1:
                st.image(media, caption="Filtro Média", use_container_width=True, channels="GRAY")
                success, buffer = cv2.imencode('.png', media)
                if success:
                    import base64
                    b64 = base64.b64encode(buffer.tobytes()).decode()
                    st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_media.png', mime='image/png')
            with col_media2:
                fig_media, ax_media = plt.subplots()
                ax_media.hist(media.ravel(), bins=256, range=[0,256])
                ax_media.set_title("Histograma Filtro Média")
                st.pyplot(fig_media)

    if st.button("Filtro Mediana"):
        mediana = filtros.apply_median_filter(img, ksize=ksize_pb)
        col_med1, col_med2 = st.columns(2)
        with col_med1:
            st.image(mediana, caption="Filtro Mediana", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', mediana)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_mediana.png', mime='image/png')
        with col_med2:
            fig_mediana, ax_mediana = plt.subplots()
            ax_mediana.hist(mediana.ravel(), bins=256, range=[0,256])
            ax_mediana.set_title("Histograma Filtro Mediana")
            st.pyplot(fig_mediana)

    if st.button("Filtro Gaussiano"):
        gauss = filtros.apply_gaussian_filter(img, ksize=ksize_pb)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(gauss, caption="Filtro Gaussiano", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', gauss)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_gaussiano.png', mime='image/png')
        with col_g2:
            fig_gauss, ax_gauss = plt.subplots()
            ax_gauss.hist(gauss.ravel(), bins=256, range=[0,256])
            ax_gauss.set_title("Histograma Filtro Gaussiano")
            st.pyplot(fig_gauss)

    if st.button("Filtro Máximo"):
        maxf = filtros.apply_max_filter(img, ksize=ksize_pb)
        col_max1, col_max2 = st.columns(2)
        with col_max1:
            st.image(maxf, caption="Filtro Máximo (Max Filter)", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', maxf)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_maximo.png', mime='image/png')
        with col_max2:
            fig_max, ax_max = plt.subplots()
            ax_max.hist(maxf.ravel(), bins=256, range=[0,256])
            ax_max.set_title("Histograma Filtro Máximo")
            st.pyplot(fig_max)

    if st.button("Filtro Mínimo"):
        minf = filtros.apply_min_filter(img, ksize=ksize_pb)
        col_min1, col_min2 = st.columns(2)
        with col_min1:
            st.image(minf, caption="Filtro Mínimo (Min Filter)", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', minf)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_minimo.png', mime='image/png')
        with col_min2:
            fig_min, ax_min = plt.subplots()
            ax_min.hist(minf.ravel(), bins=256, range=[0,256])
            ax_min.set_title("Histograma Filtro Mínimo")
            st.pyplot(fig_min)

    st.header("Filtros Passa-Alta")
    if st.button("Filtro Laplaciano"):
        lap = filtros.filtro_laplaciano(img) if hasattr(filtros, 'filtro_laplaciano') else filtros.apply_laplacian_filter(img)
        col_lap1, col_lap2 = st.columns(2)
        with col_lap1:
            st.image(lap, caption="Filtro Laplaciano", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', lap)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_laplaciano.png', mime='image/png')
        with col_lap2:
            fig_lap, ax_lap = plt.subplots()
            ax_lap.hist(lap.ravel(), bins=256, range=[0,256])
            ax_lap.set_title("Histograma Laplaciano")
            st.pyplot(fig_lap)

    if st.button("Filtro Roberts"):
        rob = filtros.filtro_roberts(img) if hasattr(filtros, 'filtro_roberts') else filtros.apply_roberts_filter(img)
        col_rob1, col_rob2 = st.columns(2)
        with col_rob1:
            st.image(rob, caption="Filtro Roberts", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', rob)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_roberts.png', mime='image/png')
        with col_rob2:
            fig_rob, ax_rob = plt.subplots()
            ax_rob.hist(rob.ravel(), bins=256, range=[0,256])
            ax_rob.set_title("Histograma Roberts")
            st.pyplot(fig_rob)

    if st.button("Filtro Prewitt"):
        prew = filtros.filtro_prewitt(img) if hasattr(filtros, 'filtro_prewitt') else filtros.apply_prewitt_filter(img)
        col_prew1, col_prew2 = st.columns(2)
        with col_prew1:
            st.image(prew, caption="Filtro Prewitt", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', prew)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_prewitt.png', mime='image/png')
        with col_prew2:
            fig_prew, ax_prew = plt.subplots()
            ax_prew.hist(prew.ravel(), bins=256, range=[0,256])
            ax_prew.set_title("Histograma Prewitt")
            st.pyplot(fig_prew)

    if st.button("Filtro Sobel"):
        sobel = filtros.filtro_sobel(img) if hasattr(filtros, 'filtro_sobel') else filtros.apply_sobel_filter(img)
        col_sob1, col_sob2 = st.columns(2)
        with col_sob1:
            st.image(sobel, caption="Filtro Sobel", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', sobel)
            if success:
                import base64
                b64 = base64.b64encode(buffer.tobytes()).decode()
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='filtro_sobel.png', mime='image/png')
        with col_sob2:
            fig_sobel, ax_sobel = plt.subplots()
            ax_sobel.hist(sobel.ravel(), bins=256, range=[0,256])
            ax_sobel.set_title("Histograma Sobel")
            st.pyplot(fig_sobel)

    # =====================
    # Espectro de Fourier
    # =====================
    st.header("Espectro de Fourier")
    st.markdown("Visualize o conteúdo em frequência da imagem através do espectro de magnitude da Transformada de Fourier.")
    if st.button("Mostrar Espectro de Fourier"):
        spectrum = filtros.apply_fourier_spectrum(img)
        fig, ax = plt.subplots()
        ax.imshow(spectrum, cmap='gray')
        ax.set_title("Magnitude do Espectro de Fourier")
        ax.axis('off')
        st.pyplot(fig)

    # =====================
    # Convolução no Domínio da Frequência
    # =====================
    st.header("Convolução no Domínio da Frequência")
    st.markdown("Compare a imagem original, a filtrada por média espacial, e as filtradas por passa-baixa e passa-alta no domínio da frequência.")

    radius = st.slider("Raio dos Filtros na Frequência", 5, 100, 30)

    # Imagem original
    img_original = img

    # Média espacial
    img_media = filtros.apply_mean_filter(img, ksize=3)

    # Passa-baixa na frequência
    img_passabaixa = filtros.apply_frequency_filter(img, filter_type="low", radius=radius)
    if img_passabaixa.dtype != np.uint8:
        img_passabaixa = np.clip(img_passabaixa, 0, 255).astype(np.uint8)

    # Passa-alta na frequência
    img_passaalta = filtros.apply_frequency_filter(img, filter_type="high", radius=radius)
    if img_passaalta.dtype != np.uint8:
        img_passaalta = np.clip(img_passaalta, 0, 255).astype(np.uint8)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.image(img_original, caption="Original", use_container_width=True, channels="GRAY")
    with colB:
        st.image(img_media, caption="Média Espacial (3x3)", use_container_width=True, channels="GRAY")
    with colC:
        st.image(img_passabaixa, caption=f"Passa-baixa (Freq, r={radius})", use_container_width=True, channels="GRAY")
    with colD:
        st.image(img_passaalta, caption=f"Passa-alta (Freq, r={radius})", use_container_width=True, channels="GRAY")

    st.header("Segmentação")
    if st.button("Otsu Threshold"):
        otsu = filtros.apply_otsu_threshold(img)
        col_otsu1, col_otsu2 = st.columns(2)
        with col_otsu1:
            st.image(otsu, caption="Segmentação Otsu", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', otsu)
            if success:
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='otsu.png', mime='image/png')
        with col_otsu2:
            fig_otsu, ax_otsu = plt.subplots()
            ax_otsu.hist(otsu.ravel(), bins=256, range=[0,256])
            ax_otsu.set_title("Histograma Otsu")
            st.pyplot(fig_otsu)

    st.header("Morfologia Matemática")

    min_dim_morf = min(img.shape[0], img.shape[1])
    st.markdown("<b>Escolha o tamanho da máscara morfológica (valor inteiro ímpar, ex: 3 para 3x3):</b>", unsafe_allow_html=True)
    ksize_morf = st.number_input("Tamanho da máscara (kernel) - Morfologia", min_value=1, max_value=min(99, min_dim_morf), value=3, step=2, format="%d")
    if ksize_morf % 2 == 0:
        st.error("O tamanho da máscara deve ser um número ímpar.")
    elif ksize_morf > min_dim_morf:
        st.error(f"O tamanho da máscara ({ksize_morf}) não pode ser maior que a menor dimensão da imagem ({min_dim_morf}).")

    
    iterations = st.slider("Iterações", 1, 5, 1)
    if st.button("Erosão"):
        erode = filtros.apply_erosion(img, ksize=ksize_morf, iterations=iterations)
        col_erode1, col_erode2 = st.columns(2)
        with col_erode1:
            st.image(erode, caption=f"Erosão (kernel={ksize_morf}, iterações={iterations})", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', erode)
            if success:
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='erosao.png', mime='image/png')
        with col_erode2:
            fig_erode, ax_erode = plt.subplots()
            ax_erode.hist(erode.ravel(), bins=256, range=[0,256])
            ax_erode.set_title("Histograma Erosão")
            st.pyplot(fig_erode)
    if st.button("Dilatação"):
        dilate = filtros.apply_dilation(img, ksize=ksize_morf, iterations=iterations)
        col_dilate1, col_dilate2 = st.columns(2)
        with col_dilate1:
            st.image(dilate, caption=f"Dilatação (kernel={ksize_morf}, iterações={iterations})", use_container_width=True, channels="GRAY")
            success, buffer = cv2.imencode('.png', dilate)
            if success:
                st.download_button(label='Salvar imagem', data=buffer.tobytes(), file_name='dilatacao.png', mime='image/png')
        with col_dilate2:
            fig_dilate, ax_dilate = plt.subplots()
            ax_dilate.hist(dilate.ravel(), bins=256, range=[0,256])
            ax_dilate.set_title("Histograma Dilatação")
            st.pyplot(fig_dilate)

    # =====================
    # DESCRITORES
    # =====================
    st.header("Descritores de Imagem")
    from descritores import shape_descriptor, color_descriptor, texture_descriptor

    # Forma
    st.markdown("## Descritor de Forma (Momentos de Hu)")
    hu, thresh = shape_descriptor(img, return_thresh=True)
    st.image(thresh, caption="Imagem Binarizada para Momentos de Hu", use_container_width=False, width=220, channels="GRAY")
    st.markdown("""
    <span style='font-size:1em'>Invariantes a rotação, escala e translação.</span>
    """, unsafe_allow_html=True)

    # Cor
    st.markdown("## Descritor de Cor (Histograma Normalizado)")
    color_hist = color_descriptor(img)
    fig_color, ax_color = plt.subplots(figsize=(4,2.2))
    colors = ['blue']*32 + ['green']*32 + ['red']*32
    ax_color.bar(np.arange(96), color_hist, color=colors, alpha=0.7)
    ax_color.set_title("Histograma de Cor (cada barra = cor do canal)")
    ax_color.set_xlabel("Bin")
    ax_color.set_ylabel("Frequência Normalizada")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='R'), Patch(facecolor='green', label='G'), Patch(facecolor='blue', label='B')]
    ax_color.legend(handles=legend_elements)
    st.pyplot(fig_color)
    st.markdown("""
    <span style='font-size:1em'>Vetor de 96 valores (32 bins para cada canal B, G, R).</span>
    <br><br>
    <b>Como interpretar este histograma de cor:</b><br>
    - Cada barra representa a frequência de pixels em uma faixa de intensidade para cada canal (azul, verde, vermelho).<br>
    - Distribuição concentrada em poucos bins indica pouca variação de cor.<br>
    - Distribuição espalhada indica riqueza de cores ou imagem colorida.<br>
    - Picos em um canal mostram predominância daquela cor.<br>
    - Imagens em tons de cinza terão os três canais similares.<br>
    """, unsafe_allow_html=True)

    # Textura
    st.markdown("## Descritor de Textura (LBP)")
    texture_hist = texture_descriptor(img)
    fig_lbp, ax_lbp = plt.subplots(figsize=(4,2.2))
    ax_lbp.bar(np.arange(10), texture_hist, color='gray', alpha=0.7)
    ax_lbp.set_title("Histograma LBP (Textura)")
    ax_lbp.set_xlabel("Padrão LBP (bin)")
    ax_lbp.set_ylabel("Frequência Normalizada")
    st.pyplot(fig_lbp)
    st.markdown("""
    <span style='font-size:1em'>Histograma de padrões locais de textura (10 bins, método 'uniform').</span>
    <br><br>
    <b>Como interpretar este histograma de textura (LBP):</b><br>
    - Cada barra representa a frequência de um padrão local de textura.<br>
    - Pico no último bin (bin 8 ou 9) geralmente indica regiões lisas (pouca textura).<br>
    - Vários bins ocupados indicam riqueza de detalhes/textura.<br>
    - Distribuição espalhada pode indicar ruído ou diversidade de texturas.<br>
    """, unsafe_allow_html=True)
