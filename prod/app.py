import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from utils import (
    load_model, 
    predict_character, 
    load_reference_embeddings,
    validate_model_and_embeddings,
    debug_embeddings_file,
    debug_model_architecture,
    CLASSES,
    idx_to_class
)
import traceback
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Personajes de Los Simpsons",
    page_icon="🟡",
    layout="centered"
)

# Función para obtener avatar del personaje
def get_character_avatar(character_name):
    """Devuelve la ruta del avatar del personaje"""
    avatar_dir = "prod/avatares"
    avatar_path = os.path.join(avatar_dir, f"{character_name}.png")
    
    # Verificar diferentes extensiones
    for ext in ['.png', '.jpg', '.jpeg']:
        test_path = os.path.join(avatar_dir, f"{character_name}{ext}")
        if os.path.exists(test_path):
            return test_path
    
    return None

# Función para cargar el modelo con manejo de errores mejorado
@st.cache_resource
def load_cached_model():
    """Carga el modelo una sola vez y lo mantiene en caché"""
    try:
        # Cargar modelo
        model_path = 'prod/modelo.pth'
        if not os.path.exists(model_path):
            return None, None, None, f"❌ Archivo de modelo no encontrado: {model_path}"
        
        # Cargar con parámetros correctos
        model, loaded_idx_to_class = load_model(
            model_path, 
            backbone='densenet121',  # Usar el backbone correcto
            embedding_size=128,
            num_classes=len(CLASSES)  # Número de clases
        )
        
        if model is None:
            return None, None, None, "❌ Error al cargar el modelo"
        
        model.eval()
        
        # Cargar embeddings de referencia
        embeddings_path = 'prod/reference_embeddings.pt'
        
        # Debug: inspeccionar archivo de embeddings
        if os.path.exists(embeddings_path):
            #st.info("🔍 Inspeccionando archivo de embeddings...")
            debug_data = debug_embeddings_file(embeddings_path)
            
            reference_embeddings = load_reference_embeddings(embeddings_path)
        else:
            st.warning("⚠️ Archivo de embeddings no encontrado. Creando embeddings vacíos...")
            reference_embeddings = None
        
        # Validar compatibilidad
        is_valid = validate_model_and_embeddings(model, reference_embeddings)
        
        if is_valid:
            print("✅ Modelo y embeddings cargados correctamente")
            #st.success("✅ Modelo y embeddings cargados correctamente")
        else:
            st.warning("⚠️ Modelo cargado con advertencias - puede haber incompatibilidades")
        
        return model, loaded_idx_to_class, reference_embeddings, None
        
    except Exception as e:
        error_msg = f"Error al cargar el modelo: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg

# Cargar el modelo
model, loaded_idx_to_class, reference_embeddings, error_msg = load_cached_model()

# Usar el idx_to_class cargado o el por defecto
current_idx_to_class = loaded_idx_to_class if loaded_idx_to_class else idx_to_class

# Transformación de imágenes (debe coincidir con el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiado a 224x224 como en utils.py
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Interfaz de usuario
st.title("🟡 Detector de Personajes de Los Simpsons")
st.markdown("### Utilizando pérdida de la trilliza y embeddings")

# Mostrar error si el modelo no se pudo cargar
if error_msg:
    st.error("❌ Error al cargar el modelo:")
    st.code(error_msg)
    st.markdown("### Posibles soluciones:")
    st.markdown("""
    1. **Verificar que el archivo `prod/modelo.pth` existe y es accesible**
    2. **Comprobar que el modelo fue guardado correctamente durante el entrenamiento**
    3. **Asegurarse de que la arquitectura del modelo coincide con la del entrenamiento**
    4. **Verificar que el archivo `prod/reference_embeddings.pt` existe**
    """)
    st.stop()

# Información sobre el modelo
st.info("📝 Sube una imagen de un personaje de Los Simpsons y el modelo identificará el personaje usando embeddings.")

# Mostrar información técnica en un expander
with st.expander("🔧 Información técnica"):
    if model is not None:
        # Test del modelo
        try:
            dummy_input = torch.randn(1, 3, 224, 224)  # Cambiado a 224x224
            with torch.no_grad():
                if hasattr(model, 'forward_normalized'):
                    output = model.forward_normalized(dummy_input)
                else:
                    output = model(dummy_input)
            st.success(f"✅ Modelo funcionando")
            
            
                
        except Exception as e:
            st.error(f"❌ Error en test del modelo: {e}")
    
    if reference_embeddings is not None:
        if isinstance(reference_embeddings, dict):
            st.success(f"✅ Embeddings de referencia (dict): {len(reference_embeddings)} personajes")
        elif torch.is_tensor(reference_embeddings):
            st.success(f"✅ Embeddings de referencia (tensor): {reference_embeddings.shape}")
        else:
            st.success(f"✅ Embeddings de referencia: {type(reference_embeddings)}")
    else:
        st.warning("⚠️ Sin embeddings de referencia")

# Mostrar personajes disponibles con avatares
with st.expander("👥 Personajes detectables"):
    if current_idx_to_class:
        cols = st.columns(3)
        for i, (idx, character) in enumerate(current_idx_to_class.items()):
            with cols[i % 3]:
                avatar_path = get_character_avatar(character)
                if avatar_path and os.path.exists(avatar_path):
                    try:
                        avatar_img = Image.open(avatar_path)
                        st.image(avatar_img, width=50)
                    except:
                        st.write("👤")
                else:
                    st.write("👤")
                st.write(f"**{character.replace('_', ' ').title()}**")

# Subida de archivo
uploaded_file = st.file_uploader(
    "📁 Subí una imagen", 
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Mostrar imagen subida
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📸 Imagen subida", use_container_width=True)
        
        with col2:
            # Procesar imagen
            with st.spinner("🔄 Analizando imagen..."):
                # Aplicar transformaciones
                tensor = transform(image).unsqueeze(0)
                
                # Predecir usando embeddings
                result = predict_character(
                    model, 
                    tensor, 
                    current_idx_to_class, 
                    reference_embeddings,
                    threshold=0.6  # Umbral ajustable
                )
                
                if result and len(result) >= 2:
                    top_predictions = result[0]
                    query_embedding = result[1] if len(result) > 1 else None
                    
                    if top_predictions and len(top_predictions) > 0:
                        # Mostrar predicción principal
                        best_prediction = top_predictions[0]
                        best_character = best_prediction[0]
                        best_score = best_prediction[1]
                        
                        # Determinar confianza
                        if best_score > 0.8:
                            confidence_color = "green"
                            confidence_text = "Alta confianza"
                        elif best_score > 0.6:
                            confidence_color = "orange"
                            confidence_text = "Confianza media"
                        else:
                            confidence_color = "red"
                            confidence_text = "Baja confianza"
                        
                        st.success(f"🎯 **Personaje detectado:**")
                        
                        # Mostrar avatar del personaje predicho
                        avatar_path = get_character_avatar(best_character)
                        if avatar_path and os.path.exists(avatar_path):
                            try:
                                avatar_img = Image.open(avatar_path)
                                st.image(avatar_img, width=100, caption=f"{best_character.replace('_', ' ').title()}")
                            except:
                                st.markdown(f"### 👤 {best_character.replace('_', ' ').title()}")
                        else:
                            st.markdown(f"### 👤 {best_character.replace('_', ' ').title()}")
                        
                        st.markdown(f"**Similitud coseno:** {best_score:.4f}")
                        st.markdown(f"**Confianza:** :{confidence_color}[{confidence_text}]")
                        
                        # Mostrar Top 5
                        st.markdown("### 🏆 Top 5 Predicciones:")
                        for i, (character, score) in enumerate(top_predictions[:5], 1):
                            with st.container():
                                rank_col, avatar_col, name_col, score_col = st.columns([0.5, 1, 3, 1])
                                
                                with rank_col:
                                    if i == 1:
                                        st.write("🥇")
                                    elif i == 2:
                                        st.write("🥈")
                                    elif i == 3:
                                        st.write("🥉")
                                    else:
                                        st.write(f"**{i}.**")
                                
                                with avatar_col:
                                    avatar_path = get_character_avatar(character)
                                    if avatar_path and os.path.exists(avatar_path):
                                        try:
                                            avatar_img = Image.open(avatar_path)
                                            st.image(avatar_img, width=40)
                                        except:
                                            st.write("👤")
                                    else:
                                        st.write("👤")
                                
                                with name_col:
                                    st.write(f"**{character.replace('_', ' ').title()}**")
                                
                                with score_col:
                                    st.write(f"{score:.4f}")
                        
                        # Información del embedding
                        if query_embedding is not None:
                            
                            
                            # Mostrar distribución del embedding
                            if len(query_embedding.shape) > 1:
                                emb_flat = query_embedding.flatten()
                            else:
                                emb_flat = query_embedding
                    else:
                        st.error("❌ No se pudieron obtener predicciones válidas")
                else:
                    st.error("❌ Error al procesar la imagen")
        
        # Información adicional
        st.markdown("---")
        st.info("💡 **Tip:** Para mejores resultados, usa imágenes claras con el personaje bien visible y sin mucho fondo.")
        
        # Debug info
        with st.expander("🐛 Información de debug"):
            st.markdown("**Transformaciones aplicadas:**")
            st.code("""
            1. Resize a (128, 128)
            2. Conversión a Tensor
            3. Normalización ImageNet
            """)
            
            
            # Información del modelo
            if model is not None:
                st.markdown("**Información del modelo:**")
                st.markdown(f"- Tipo: {type(model).__name__}")
                if hasattr(model, 'backbone_name'):
                    st.markdown(f"- Backbone: {model.backbone_name}")
                if hasattr(model, 'embedding_size'):
                    st.markdown(f"- Embedding size: {model.embedding_size}")
        
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        st.code(traceback.format_exc())

# Sección de configuración avanzada
with st.expander("⚙️ Configuración avanzada"):
    st.markdown("**Parámetros del modelo:**")
    
    # Selector de umbral
    threshold = st.slider(
        "Umbral de confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Umbral mínimo para considerar una predicción válida"
    )
    
    # Selector de backbone (para información)
    backbone_info = st.selectbox(
        "Backbone utilizado",
        ["densenet121", "resnet18", "efficientnet_b0"],
        index=0,
        disabled=True,
        help="Arquitectura base del modelo (solo información)"
    )
    
    # Información de embeddings
    if reference_embeddings is not None:
        st.markdown("**Embeddings de referencia:**")
        if isinstance(reference_embeddings, dict):
            st.markdown(f"- Formato: Diccionario")
            st.markdown(f"- Personajes: {len(reference_embeddings)}")
            sample_char = list(reference_embeddings.keys())[0]
            sample_emb = reference_embeddings[sample_char]
            

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Desarrollado para el Trabajo Práctico Integrador de<br>
        Redes Neuronales Profundas - Ingeniería en Sistemas de Información<br>
        
    </div>
    """, 
    unsafe_allow_html=True
)