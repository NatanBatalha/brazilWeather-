import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.set_option("display.max_columns", 100)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)


def main():
    print(">>> Iniciando análise do clima (brazilWeather)")

    # Pastas
    data_folder = "./data"
    output_folder = "./outputs"
    plots_folder = "./plots"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Listar arquivos disponíveis
    # ------------------------------------------------------------------
    print(f"Caminho da pasta de dados: {os.path.abspath(data_folder)}")

    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(">>> Arquivos encontrados:")
    for f in all_files:
        print(" -", f)

    print("Quantidade de arquivos CSV encontrados:", len(all_files))

    codes_file = None
    weather_file = None

    for f in all_files:
        nome = os.path.basename(f).lower()
        if "codes" in nome:
            codes_file = f
        elif "weather" in nome:
            weather_file = f

    print("\nArquivo de CÓDIGOS de estação:", codes_file)
    print("Arquivo de DADOS de clima    :", weather_file)

    if codes_file is None or weather_file is None:
        print("!!! Não consegui identificar os dois arquivos necessários (codes e weather).")
        return

    # ------------------------------------------------------------------
    # 2. Ler df_stations (estações) com sep=';'
    # ------------------------------------------------------------------
    df_stations = pd.read_csv(codes_file, sep=";")
    print("\n>>> df_stations (estações) lido com sucesso!")
    print("Formato:", df_stations.shape)
    print(df_stations.head())
    print("\nColunas de df_stations:", list(df_stations.columns))

    # Renomear colunas para nomes mais amigáveis
    df_stations = df_stations.rename(
        columns={
            "REGIAO": "regiao",
            "UF": "uf",
            "ESTACAO": "nome_estacao",
            "CODIGO": "estacao",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "ALTITUDE": "altitude_m",
        }
    )

    print("\nColunas renomeadas de df_stations:", list(df_stations.columns))

    # ------------------------------------------------------------------
    # 3. Ler uma AMOSTRA de df_raw (dados climáticos) com sep=';'
    # ------------------------------------------------------------------
    print("\n>>> Lendo AMOSTRA do arquivo de dados climáticos (100k linhas)...")
    df_raw = pd.read_csv(
        weather_file,
        sep=";",
        nrows=100_000,      # você pode aumentar/diminuir se quiser
        low_memory=False,
    )
    print(">>> df_raw lido com sucesso!")
    print("Formato da amostra:", df_raw.shape)
    print("\nPrimeiras linhas de df_raw:")
    print(df_raw.head())
    print("\nColunas de df_raw:", list(df_raw.columns))

    # Renomear colunas do clima
    df_raw = df_raw.rename(
        columns={
            "ESTACAO": "estacao",
            "DATA (YYYY-MM-DD)": "data",
            "HORA (UTC)": "hora_utc",
            "PRECIPITACAO TOTAL HORARIO (mm)": "precipitacao_mm",
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": "pressao_estacao_mb",
            "PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": "pressao_max_mb",
            "PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": "pressao_min_mb",
            "RADIACAO GLOBAL (W/m2)": "radiacao_wm2",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (C)": "temp_bulbo_seco_c",
            "TEMPERATURA DO PONTO DE ORVALHO (C)": "temp_orvalho_c",
            "TEMPERATURA MAXIMA NA HORA ANT. (AUT) (C)": "temp_max_c",
            "TEMPERATURA MINIMA NA HORA ANT. (AUT) (C)": "temp_min_c",
            "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (C)": "temp_orvalho_max_c",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (C)": "temp_orvalho_min_c",
            "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": "umid_max_pct",
            "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": "umid_min_pct",
            "UMIDADE RELATIVA DO AR, HORARIA (%)": "umid_pct",
            "VENTO, DIRECAO HORARIA (gr)": "vento_dir_graus",
            "VENTO, RAJADA MAXIMA (m/s)": "vento_rajada_ms",
            "VENTO, VELOCIDADE HORARIA (m/s)": "vento_vel_ms",
        }
    )

    print("\nColunas renomeadas de df_raw:", list(df_raw.columns))

    # ------------------------------------------------------------------
    # 4. Merge clima + estações
    # ------------------------------------------------------------------
    df = df_raw.merge(df_stations, on="estacao", how="left")
    print("\n>>> df (clima + info estação) após merge:")
    print("Formato:", df.shape)
    print(df[["estacao", "uf", "regiao"]].head())

    # ------------------------------------------------------------------
    # 5. Engenharia de atributos de data
    # ------------------------------------------------------------------
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day

    # Converter hora_utc para numérico (pode vir como string)
    df["hora_utc"] = pd.to_numeric(df["hora_utc"], errors="coerce")

    def estacao_do_ano(mes):
        if pd.isna(mes):
            return pd.NA
        mes = int(mes)
        if mes in (12, 1, 2):
            return "verao"
        if mes in (3, 4, 5):
            return "outono"
        if mes in (6, 7, 8):
            return "inverno"
        if mes in (9, 10, 11):
            return "primavera"
        return pd.NA

    df["estacao_ano"] = df["mes"].apply(estacao_do_ano)

    print("\nExemplo de colunas de data e estacao_ano:")
    print(df[["data", "ano", "mes", "dia", "hora_utc", "estacao_ano"]].head())

    # ------------------------------------------------------------------
    # 6. Variável alvo: choveu (classificação)
    # ------------------------------------------------------------------
    df["precipitacao_mm"] = pd.to_numeric(df["precipitacao_mm"], errors="coerce")
    df["choveu"] = (df["precipitacao_mm"] > 0).astype(int)

    print("\nDistribuição de choveu (0=sem chuva, 1=com chuva):")
    print(df["choveu"].value_counts())
    print("\nProporções de choveu:")
    print(df["choveu"].value_counts(normalize=True))

    # Salvar uma amostra preparada para usar no Power BI
    cols_export = [
        "data",
        "hora_utc",
        "uf",
        "regiao",
        "estacao_ano",
        "latitude",
        "longitude",
        "altitude_m",
        "precipitacao_mm",
        "choveu",
        "temp_bulbo_seco_c",
        "temp_max_c",
        "temp_min_c",
        "temp_orvalho_c",
        "umid_pct",
        "umid_max_pct",
        "umid_min_pct",
        "pressao_estacao_mb",
        "radiacao_wm2",
        "vento_vel_ms",
        "vento_rajada_ms",
    ]
    cols_export = [c for c in cols_export if c in df.columns]
    df_export = df[cols_export].copy()
    export_path = os.path.join(output_folder, "df_amostra_trabalho.csv")
    df_export.to_csv(export_path, index=False)
    print(f"\nArquivo para Power BI salvo em: {export_path}")

    # ------------------------------------------------------------------
    # 7. Análise Univariada (10 variáveis)
    # ------------------------------------------------------------------
    print("\n=== ANÁLISE UNIVARIADA ===")

    univ_candidates = [
        "temp_bulbo_seco_c",
        "temp_max_c",
        "temp_min_c",
        "temp_orvalho_c",
        "umid_pct",
        "umid_max_pct",
        "umid_min_pct",
        "pressao_estacao_mb",
        "radiacao_wm2",
        "vento_vel_ms",
        "altitude_m",
    ]

    univ_vars = [c for c in univ_candidates if c in df.columns]
    print("Variáveis selecionadas para univariada:", univ_vars)

    univ_rows = []

    for col in univ_vars:
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.empty:
            continue

        moda = serie.mode()
        moda_val = moda.iloc[0] if not moda.empty else np.nan

        row = {
            "variavel": col,
            "media": serie.mean(),
            "mediana": serie.median(),
            "moda": moda_val,
            "desvio_padrao": serie.std(),
            "p25": serie.quantile(0.25),
            "p50": serie.quantile(0.50),
            "p75": serie.quantile(0.75),
        }
        univ_rows.append(row)

        # Histograma salvo em PNG
        plt.figure()
        sns.histplot(serie, kde=True)
        plt.title(f"Histograma - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()
        hist_path = os.path.join(plots_folder, f"hist_{col}.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Histograma salvo: {hist_path}")

    df_univ = pd.DataFrame(univ_rows)
    univ_path = os.path.join(output_folder, "univariada_resumo.csv")
    df_univ.to_csv(univ_path, index=False)
    print("\nResumo univariado salvo em:", univ_path)
    print(df_univ)

    # ------------------------------------------------------------------
    # 8. Análise Multivariada - Correlação de Pearson
    # ------------------------------------------------------------------
    print("\n=== ANÁLISE MULTIVARIADA (CORRELAÇÃO DE PEARSON) ===")

    # Converter categóricas relevantes em códigos numéricos
    df_corr = df.copy()
    for cat_col in ["uf", "regiao", "estacao_ano"]:
        if cat_col in df_corr.columns:
            df_corr[cat_col + "_cod"] = df_corr[cat_col].astype("category").cat.codes

    numeric_cols = df_corr.select_dtypes(include=["number"]).columns
    corr = df_corr[numeric_cols].corr(method="pearson")

    corr_path = os.path.join(output_folder, "correlacao_pearson.csv")
    corr.to_csv(corr_path)
    print("Matriz de correlação salva em:", corr_path)

    if "choveu" in corr.columns:
        print("\nCorrelação de Pearson com a variável alvo 'choveu':")
        print(corr["choveu"].sort_values(ascending=False))

    # Heatmap de correlação
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matriz de Correlação de Pearson")
    plt.tight_layout()
    heatmap_path = os.path.join(plots_folder, "correlacao_pearson_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print("Heatmap de correlação salvo em:", heatmap_path)

    # ------------------------------------------------------------------
    # 9. Visualizações (exploratória e explicativa)
    # ------------------------------------------------------------------
    print("\n=== VISUALIZAÇÕES ===")

    # Amostra para gráficos (pra não pesar)
    if len(df) > 20_000:
        df_plot = df.sample(20_000, random_state=42)
    else:
        df_plot = df.copy()

    # 9.1 Boxplot de precipitação por estação do ano (exploratória)
    if "estacao_ano" in df_plot.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_plot, x="estacao_ano", y="precipitacao_mm")
        plt.title("Precipitação por Estação do Ano")
        plt.xlabel("Estação do ano")
        plt.ylabel("Precipitação (mm)")
        plt.tight_layout()
        box_path = os.path.join(plots_folder, "boxplot_precipitacao_por_estacao_ano.png")
        plt.savefig(box_path)
        plt.close()
        print("Gráfico salvo:", box_path)

    # 9.2 Proporção de registros com chuva por estação do ano (explicativa)
    if "choveu" in df_plot.columns and "estacao_ano" in df_plot.columns:
        chuva_por_estacao = df_plot.groupby("estacao_ano")["choveu"].mean().sort_index()
        plt.figure(figsize=(8, 5))
        chuva_por_estacao.plot(kind="bar")
        plt.title("Proporção de registros com chuva por estação do ano")
        plt.xlabel("Estação do ano")
        plt.ylabel("Proporção de choveu")
        plt.tight_layout()
        bar_path = os.path.join(plots_folder, "barra_choveu_por_estacao_ano.png")
        plt.savefig(bar_path)
        plt.close()
        print("Gráfico salvo:", bar_path)

    # 9.3 Temperatura média por hora do dia (explicativa)
    if "hora_utc" in df_plot.columns and "temp_bulbo_seco_c" in df_plot.columns:
        temp_por_hora = (
            df_plot.groupby("hora_utc")["temp_bulbo_seco_c"]
            .mean()
            .sort_index()
        )
        plt.figure(figsize=(8, 5))
        temp_por_hora.plot(kind="line", marker="o")
        plt.title("Temperatura média por hora (UTC)")
        plt.xlabel("Hora (UTC)")
        plt.ylabel("Temperatura do ar (°C)")
        plt.tight_layout()
        line_path = os.path.join(plots_folder, "linha_temp_media_por_hora.png")
        plt.savefig(line_path)
        plt.close()
        print("Gráfico salvo:", line_path)

    # ------------------------------------------------------------------
    # 10. Modelos de Classificação (ML)
    # ------------------------------------------------------------------
    print("\n=== MODELOS DE CLASSIFICAÇÃO (ML) ===")

    if "choveu" not in df.columns:
        print("Não há coluna 'choveu' no dataframe. Não é possível treinar modelos.")
        return

    df_ml = df.copy()

    # Codificar categóricas
    cat_cols = []
    for cat in ["uf", "regiao", "estacao_ano"]:
        if cat in df_ml.columns:
            code_col = cat + "_cod"
            df_ml[code_col] = df_ml[cat].astype("category").cat.codes
            cat_cols.append(code_col)

    # Selecionar features numéricas relevantes
    num_features_candidates = [
        "temp_bulbo_seco_c",
        "temp_max_c",
        "temp_min_c",
        "temp_orvalho_c",
        "umid_pct",
        "umid_max_pct",
        "umid_min_pct",
        "pressao_estacao_mb",
        "radiacao_wm2",
        "vento_vel_ms",
        "vento_rajada_ms",
        "hora_utc",
        "mes",
        "ano",
        "altitude_m",
        "latitude",
        "longitude",
    ]
    num_features = [c for c in num_features_candidates if c in df_ml.columns]

    feature_cols = num_features + cat_cols
    print("Features usadas no ML:", feature_cols)

    X = df_ml[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df_ml["choveu"]

    # Remover linhas onde y é NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Tratar NaNs em X com mediana
    X = X.fillna(X.median(numeric_only=True))

    # Amostra opcional para acelerar (por ex., 50k registros)
    if len(X) > 50_000:
        X_sample = X.sample(50_000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
    )

    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }

    resultados_ml = []

    for nome_modelo, modelo in modelos.items():
        print(f"\n--- Treinando modelo: {nome_modelo} ---")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de confusão:")
        print(cm)

        resultados_ml.append({"modelo": nome_modelo, "acuracia": acc})

    df_resultados_ml = pd.DataFrame(resultados_ml)
    ml_path = os.path.join(output_folder, "resultados_modelos_ml.csv")
    df_resultados_ml.to_csv(ml_path, index=False)
    print("\nResumo dos modelos salvo em:", ml_path)
    print(df_resultados_ml)

    print("\n>>> Fim da execução do analise_clima.py")


if __name__ == "__main__":
    main()
