import argparse
import ROOT
import pandas as pd

def analyze_root():
    # Configuración de argparse
    parser = argparse.ArgumentParser(description="Análisis de RDataFrame con Pandas")
    parser.add_argument("--input", type=str, required=True, help="Ruta al archivo .root")
    parser.add_argument("--tree", type=str, default="events", help="Nombre del TTree")
    
    args = parser.parse_args()

    # 1. Crear RDataFrame desde el archivo proporcionado
    # La sintaxis es ROOT.RDataFrame(tree_name, file_path)
    rdf = ROOT.RDataFrame(args.tree, args.input)

    # 2. Convertir a diccionario de NumPy y luego a Pandas
    # AsNumpy() es la vía más directa para puentear ROOT y el ecosistema de Python
    data_dict = rdf.AsNumpy()
    df = pd.DataFrame(data_dict)

    # 3. Visualización de metadatos y datos
    print(f"--- Análisis del archivo: {args.input} ---")
    print(f"Columnas detectadas: {df.columns.tolist()}")
    print("\nPrimeras 5 filas:")
    print(df.head())

    # Información técnica del DataFrame
    print("\nResumen técnico:")
    print(df.info())

if __name__ == "__main__":
    analyze_root()
