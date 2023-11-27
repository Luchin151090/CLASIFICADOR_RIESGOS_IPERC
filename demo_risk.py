from transformers import BertForSequenceClassification, BertTokenizer
import torch #LIBRERIA
# Cargar el modelo pre-entrenado
model = BertForSequenceClassification.from_pretrained("C:\\Users\\luchi\\OneDrive\\Escritorio\\model_auditoria")

# Inicializa el tokenizador
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")


# Hacer inferencias en nuevos datos
input_text = "Desconexión de sistemas críticos por errores de configuración" # COMENTARIO DE TEST
input_tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
output = model(**input_tokens)
predicted_label = torch.argmax(output.logits, dim=1)

print(f"Texto: {input_text}")
print(f"Etiqueta predicha: {predicted_label}")
