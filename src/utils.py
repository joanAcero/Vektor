import os
import glob


def clean_data_folder(folder_path='data'):
    """Elimina todos los archivos .csv de la carpeta especificada."""
    # Busca todos los archivos que terminen en .csv dentro de la carpeta
    csv_files = glob.glob(f"{folder_path}/*.csv")

    if not csv_files:
        print("\n🧹 No hay archivos CSV para limpiar.")
        return

    print(f"\n🧹 Limpiando caché ({len(csv_files)} archivos)...")
    for file_path in csv_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"⚠️ Error eliminando {file_path}: {e}")

    print("✅ Carpeta data limpia.")

def get_input_or_default(prompt, default_value, type_func):
    """
    Solicita input al usuario. Si está vacío, devuelve el default.
    Si el formato es incorrecto, avisa y usa el default.
    """
    user_input = input(f"🔹 {prompt} [Default: {default_value}]: ").strip()

    if not user_input:
        return default_value

    try:
        return type_func(user_input)
    except ValueError:
        print(f"   ⚠️ Valor inválido. Usando default: {default_value}")
        return default_value
