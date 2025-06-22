import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
#import cv2
import os

# Lista de clases (labels)
CLASSES = [
    'abraham_grampa_simpson',
    'agnes_skinner',
    'apu_nahasapeemapetilon',
    'bart_simpson',
    'barney_gumble',
    'carl_carlson',
    'charles_montgomery_burns',
    'chief_wiggum',
    'cletus_spuckler',
    'comic_book_guy',
    'disco_stu',
    'edna_krabappel',
    'fat_tony',
    'gil',
    'groundskeeper_willie',
    'homer_simpson',
    'kent_brockman',
    'krusty_the_clown',
    'lenny_leonard',
    'lisa_simpson',
    'lionel_hutz',
    'maggie_simpson',
    'marge_simpson',
    'martin_prince',
    'mayor_quimby',
    'milhouse_van_houten',
    'miss_hoover',
    'moe_szyslak',
    'nelson_muntz',
    'ned_flanders',
    'otto_mann',
    'patty_bouvier',
    'principal_skinner',
    'professor_john_frink',
    'ralph_wiggum',
    'rainier_wolfcastle',
    'selma_bouvier',
    'sideshow_bob',
    'sideshow_mel',
    'snake_jailbird',
    'troy_mcclure',
    'waylon_smithers'
]

# Diccionario índice a clase
idx_to_class = {i: label for i, label in enumerate(CLASSES)}

class EmbeddingNet(nn.Module):
    """
    ARQUITECTURA EXACTA que coincide con tu modelo de entrenamiento
    """
    def __init__(self, backbone='densenet121', embedding_size=128):
        super().__init__()
        self.backbone_name = backbone
        self.embedding_size = embedding_size

        if backbone == 'resnet18':
            base = models.resnet18(pretrained=True)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(pretrained=True)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        elif backbone == 'densenet121':
            base = models.densenet121(pretrained=True)
            in_features = base.classifier.in_features
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} no soportado.")

        self.backbone = base
        self.embedding = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        return self.embedding(x)

# Clase adicional para compatibilidad (si necesitas clasificación)
class EmbeddingNetWithClassifier(EmbeddingNet):
    """
    Extensión del modelo base que añade clasificación
    """
    def __init__(self, backbone='densenet121', embedding_size=128, num_classes=None):
        super().__init__(backbone, embedding_size)
        if num_classes:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = None
    
    def forward_embeddings(self, x):
        """Solo embeddings (sin normalizar)"""
        return super().forward(x)
    
    def forward_normalized(self, x):
        """Embeddings normalizados"""
        embeddings = self.forward_embeddings(x)
        return nn.functional.normalize(embeddings, p=2, dim=1)
    
    def forward_classification(self, x):
        """Embeddings + clasificación"""
        embeddings = self.forward_embeddings(x)
        if self.classifier:
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings, None
    
    def forward(self, x):
        """Forward por defecto: embeddings normalizados"""
        return self.forward_normalized(x)

def load_model(model_path, backbone='densenet121', embedding_size=128, num_classes=None):
    """
    Carga el modelo entrenado con la arquitectura correcta
    """
    try:
        print(f"🔄 Cargando modelo desde: {model_path}")
        
        # Crear el modelo con la arquitectura EXACTA de entrenamiento
        if num_classes:
            model = EmbeddingNetWithClassifier(backbone, embedding_size, num_classes)
        else:
            model = EmbeddingNet(backbone, embedding_size)
        
        # Cargar los pesos
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Para versiones más antiguas de PyTorch
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Manejar diferentes formatos de checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📦 Checkpoint con 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("📦 Checkpoint con 'state_dict'")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("📦 Checkpoint con 'model'")
            else:
                # Asumir que el dict completo es el state_dict
                state_dict = checkpoint
                print("📦 Checkpoint directo como state_dict")
                
            # Mostrar información adicional si está disponible
            if 'epoch' in checkpoint:
                print(f"📊 Época: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"📊 Loss: {checkpoint['loss']:.4f}")
            if 'accuracy' in checkpoint:
                print(f"📊 Accuracy: {checkpoint['accuracy']:.4f}")
                
        else:
            # El checkpoint es directamente el state_dict
            state_dict = checkpoint
            print("📦 State dict directo")
        
        # Mostrar las claves del modelo para debug
        print(f"🔑 Claves en state_dict: {len(state_dict)} encontradas")
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Verificar compatibilidad
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"⚠️ Claves faltantes en checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Claves inesperadas en checkpoint: {unexpected_keys}")
        
        # Cargar los pesos
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Modelo cargado correctamente")
        print(f"🏗️ Arquitectura: {backbone}")
        print(f"📏 Dimensión embeddings: {embedding_size}")
        
        return model, idx_to_class
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        
        # Crear modelo vacío como fallback
        print("⚠️ Creando modelo vacío como fallback...")
        if num_classes:
            model = EmbeddingNetWithClassifier(backbone, embedding_size, num_classes)
        else:
            model = EmbeddingNet(backbone, embedding_size)
        return model, idx_to_class

def load_reference_embeddings(embeddings_path):
    """
    Carga los embeddings de referencia desde un archivo .pt
    """
    try:
        print(f"🔄 Cargando embeddings desde: {embeddings_path}")
        
        # Cargar datos
        try:
            data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(embeddings_path, map_location='cpu')
        
        print(f"📊 Tipo de datos: {type(data)}")
        
        # Procesar según el formato
        if isinstance(data, dict):
            print(f"📚 Diccionario con {len(data)} entradas")
            
            # Verificar si las claves coinciden con nuestros personajes
            sample_keys = list(data.keys())[:3]
            print(f"🔍 Claves ejemplo: {sample_keys}")
            
            return data
            
        elif isinstance(data, (list, tuple)):
            print(f"📝 Lista/Tuple con {len(data)} elementos")
            
            # Buscar tensor de embeddings
            for i, item in enumerate(data):
                if torch.is_tensor(item) and len(item.shape) == 2:
                    print(f"✅ Embeddings encontrados en posición {i}: {item.shape}")
                    return item
                elif isinstance(item, np.ndarray) and len(item.shape) == 2:
                    print(f"✅ Embeddings numpy encontrados en posición {i}: {item.shape}")
                    return torch.tensor(item, dtype=torch.float32)
            
            print("❌ No se encontraron embeddings válidos en la lista")
            return None
            
        elif torch.is_tensor(data):
            print(f"✅ Tensor directo: {data.shape}")
            return data
            
        elif isinstance(data, np.ndarray):
            print(f"✅ NumPy array: {data.shape}")
            return torch.tensor(data, dtype=torch.float32)
            
        else:
            print(f"❌ Formato no reconocido: {type(data)}")
            return None
            
    except Exception as e:
        print(f"❌ Error cargando embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesa una imagen para el modelo
    """
    try:
        # Cargar imagen
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Transformaciones (las mismas que usaste en entrenamiento)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor
    
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        return None

def extract_embedding(model, image_tensor):
    """
    Extrae el embedding de una imagen usando el modelo
    """
    try:
        model.eval()
        with torch.no_grad():
            # Usar el método correcto según el tipo de modelo
            if hasattr(model, 'forward_normalized'):
                embedding = model.forward_normalized(image_tensor)
            else:
                embedding = model(image_tensor)
                # Normalizar manualmente si no está normalizado
                embedding = nn.functional.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()
    except Exception as e:
        print(f"❌ Error extrayendo embedding: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """
    Calcula la similitud coseno entre dos embeddings
    """
    try:
        # Asegurar que son arrays numpy
        if torch.is_tensor(embedding1):
            embedding1 = embedding1.cpu().numpy()
        if torch.is_tensor(embedding2):
            embedding2 = embedding2.cpu().numpy()
        
        # Flatten para asegurar 1D
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        
        # Calcular normas
        norm1 = np.linalg.norm(emb1_flat)
        norm2 = np.linalg.norm(emb2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Similitud coseno
        similarity = np.dot(emb1_flat, emb2_flat) / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))  # Clip para evitar errores numéricos
    
    except Exception as e:
        print(f"❌ Error calculando similitud: {e}")
        return 0.0

def predict_character(model, img_tensor, idx_to_class, reference_embeddings=None, threshold=0.7):
    """
    Predice el personaje usando embeddings y similitud coseno
    """
    try:
        model.eval()
        with torch.no_grad():
            # Obtener embedding de la imagen
            query_embedding = extract_embedding(model, img_tensor)
            if query_embedding is None:
                return None, None
            
            if reference_embeddings is None:
                print("⚠️ No hay embeddings de referencia")
                return None, query_embedding
            
            # Convertir dict a formato compatible
            if isinstance(reference_embeddings, dict):
                similarities = {}
                for char_name, ref_emb in reference_embeddings.items():
                    sim = calculate_similarity(query_embedding, ref_emb)
                    similarities[char_name] = sim
                
                # Crear lista de predicciones
                predictions = [(char, sim) for char, sim in similarities.items()]
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                return predictions, query_embedding
            
            elif torch.is_tensor(reference_embeddings) or isinstance(reference_embeddings, np.ndarray):
                # Formato tensor/array
                if torch.is_tensor(reference_embeddings):
                    ref_embeddings = reference_embeddings.cpu().numpy()
                else:
                    ref_embeddings = reference_embeddings
                
                predictions = []
                for i, ref_emb in enumerate(ref_embeddings):
                    char_name = idx_to_class.get(i, f"Character_{i}")
                    sim = calculate_similarity(query_embedding, ref_emb)
                    predictions.append((char_name, sim))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions, query_embedding
            
            else:
                print(f"❌ Formato de embeddings no soportado: {type(reference_embeddings)}")
                return None, query_embedding
            
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_best_match(query_embedding, reference_embeddings, threshold=0.7):
    """
    Encuentra el mejor match para un embedding
    """
    try:
        if isinstance(reference_embeddings, dict):
            similarities = {}
            for char_name, ref_emb in reference_embeddings.items():
                sim = calculate_similarity(query_embedding, ref_emb)
                similarities[char_name] = sim
        
        elif torch.is_tensor(reference_embeddings) or isinstance(reference_embeddings, np.ndarray):
            similarities = {}
            if torch.is_tensor(reference_embeddings):
                ref_embeddings = reference_embeddings.cpu().numpy()
            else:
                ref_embeddings = reference_embeddings
            
            for i, ref_emb in enumerate(ref_embeddings):
                char_name = idx_to_class.get(i, f"Character_{i}")
                sim = calculate_similarity(query_embedding, ref_emb)
                similarities[char_name] = sim
        
        else:
            return None, 0.0, {}
        
        if not similarities:
            return None, 0.0, {}
        
        # Mejor match
        best_character = max(similarities, key=similarities.get)
        best_similarity = similarities[best_character]
        
        if best_similarity >= threshold:
            return best_character, best_similarity, similarities
        else:
            return None, best_similarity, similarities
    
    except Exception as e:
        print(f"❌ Error encontrando mejor match: {e}")
        return None, 0.0, {}

def validate_model_and_embeddings(model, reference_embeddings):
    """
    Valida compatibilidad entre modelo y embeddings
    """
    try:
        # Test del modelo
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Modelo OK - Output: {output.shape}")
        
        # Test de embeddings
        if reference_embeddings is not None:
            if isinstance(reference_embeddings, dict):
                sample_emb = next(iter(reference_embeddings.values()))
                if torch.is_tensor(sample_emb):
                    emb_dim = sample_emb.shape[-1]
                else:
                    emb_dim = sample_emb.shape[-1]
                print(f"✅ Embeddings dict OK - Dim: {emb_dim}")
            
            elif torch.is_tensor(reference_embeddings):
                emb_dim = reference_embeddings.shape[-1]
                print(f"✅ Embeddings tensor OK - Shape: {reference_embeddings.shape}")
            
            else:
                print(f"❌ Formato embeddings no válido: {type(reference_embeddings)}")
                return False
            
            # Verificar dimensiones
            if output.shape[1] != emb_dim:
                print(f"⚠️ Dimensiones no coinciden: modelo={output.shape[1]}, embeddings={emb_dim}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return False

def debug_model_architecture(model):
    """
    Muestra información detallada sobre la arquitectura del modelo
    """
    print("\n" + "="*50)
    print("🏗️ ARQUITECTURA DEL MODELO")
    print("="*50)
    
    print(f"📝 Tipo: {type(model).__name__}")
    
    if hasattr(model, 'backbone_name'):
        print(f"🦴 Backbone: {model.backbone_name}")
    if hasattr(model, 'embedding_size'):
        print(f"📏 Embedding size: {model.embedding_size}")
    
    print(f"\n📊 Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print(f"\n🔍 Capas principales:")
    for name, module in model.named_children():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module}")
        elif hasattr(module, '__len__'):
            print(f"  {name}: {len(module)} subcapas")
        else:
            print(f"  {name}: {type(module).__name__}")
    
    print("="*50)

# Funciones auxiliares para compatibilidad
def format_character_name(character_name):
    """Formatea nombre del personaje"""
    return character_name.replace('_', ' ').title()

def print_classification_results(predictions, threshold=0.7):
    """Imprime resultados de clasificación"""
    if not predictions:
        print("❌ No se pudieron obtener predicciones")
        return
    
    print("\n" + "="*60)
    print("🎭 RESULTADOS DE CLASIFICACIÓN")
    print("="*60)
    
    best_char, best_sim = predictions[0]
    
    if best_sim >= threshold:
        print(f"✅ IDENTIFICADO: {format_character_name(best_char)}")
        print(f"🎯 Confianza: {best_sim:.3f} ({best_sim*100:.1f}%)")
    else:
        print(f"❓ NO IDENTIFICADO (confianza baja)")
        print(f"🎯 Mejor match: {format_character_name(best_char)} - {best_sim:.3f} ({best_sim*100:.1f}%)")
    
    print(f"\n📊 TOP 5 CANDIDATOS:")
    print("-" * 40)
    
    for i, (char_name, similarity) in enumerate(predictions[:5], 1):
        status = "✅" if similarity >= threshold else "❌"
        formatted_name = format_character_name(char_name)
        print(f"{i}. {status} {formatted_name}: {similarity:.3f} ({similarity*100:.1f}%)")
    
    print("="*60)

def debug_embeddings_file(embeddings_path):
    """Debug del archivo de embeddings"""
    if not os.path.exists(embeddings_path):
        print(f"❌ Archivo no existe: {embeddings_path}")
        return None
    
    try:
        print(f"🔍 Archivo: {embeddings_path}")
        print(f"📁 Tamaño: {os.path.getsize(embeddings_path)} bytes")
        
        data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        print(f"🔍 Tipo: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📚 Dict con {len(data)} claves")
            sample_keys = list(data.keys())[:3]
            for key in sample_keys:
                value = data[key]
                shape = getattr(value, 'shape', 'No shape')
                print(f"  '{key}': {type(value)} - {shape}")
        
        elif isinstance(data, (list, tuple)):
            print(f"📝 {type(data).__name__} con {len(data)} elementos")
            for i, item in enumerate(data[:3]):
                shape = getattr(item, 'shape', f'len={len(item)}' if hasattr(item, '__len__') else 'No info')
                print(f"  [{i}]: {type(item)} - {shape}")
        
        elif torch.is_tensor(data):
            print(f"🎯 Tensor: {data.shape}, dtype: {data.dtype}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None