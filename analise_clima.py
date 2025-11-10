import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


def garantir_pastas(base_dir: str):
    outputs_dir = os.path.join(base_dir, "outputs")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return outputs_dir, plots_dir


def definir_estacao_ano_br(mes: int) -> str:
    if mes in (12, 1, 2):
        return "verao"
    if mes in (3, 4, 5):
        return "outono"
    if mes in (6, 7, 8):
        return "inverno"
    if mes in (9, 10, 11):
        return "primavera"
    return "desconhecida"


def carregar_estacoes(caminho_stations: str) -> pd.DataFrame:
    print("\nArquivo de CÓDIGOS de estação:", caminho_stations)
    df_stations = pd.read_csv(
        caminho_stations,
        sep=";",
        encoding="latin1",
    )
    print("\n>>> df_stations (estações) lido com sucesso!")
    print("Formato:", df_stations.shape)
    print(df_stations.head())
    print("\nColunas de df_stations:", list(df_stations.columns))

    rename_map = {
        "REGIAO": "regiao",
        "UF": "uf",
        "ESTACAO": "nome_estacao",
        "CODIGO": "estacao",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ALTITUDE": "altitude_m",
    }
    df_stations = df_stations.rename(columns=rename_map)

    # Aqui vamos garantir tipos numericos
    for col in ["latitude", "longitude", "altitude_m"]:
        df_stations[col] = pd.to_numeric(df_stations[col], errors="coerce")

    print("\nColunas renomeadas de df_stations:", list(df_stations.columns))
    return df_stations


def carregar_dados_clima_amostrado(
    caminho_dados: str,
    frac_por_chunk: float = 0.02,
    max_linhas_final: int = 150000,
) -> pd.DataFrame:
    print("\nArquivo de DADOS de clima   :", caminho_dados)
    print("\n>>> Lendo dados climáticos com amostragem distribuída ao longo de TODO o arquivo...")
    chunksize = 200_000
    reader = pd.read_csv(
        caminho_dados,
        sep=";",
        encoding="latin1",
        low_memory=False,
        chunksize=chunksize,
    )

    amostras = []
    total_amostras = 0
    for i, chunk in enumerate(reader, start=1):
        linhas_chunk = len(chunk)
        sample_n = max(1, int(linhas_chunk * frac_por_chunk))
        sample_chunk = chunk.sample(n=sample_n, random_state=42 + i)
        amostras.append(sample_chunk)
        total_amostras += sample_n
        print(f"   Chunk {i}: {linhas_chunk} linhas, amostradas {sample_n} (total amostras acumuladas: {total_amostras})")

    df_raw = pd.concat(amostras, ignore_index=True)
    print("\n>>> df_raw (amostra distribuída) lido com sucesso!")
    print("Formato da amostra antes de downsample:", df_raw.shape)
    print("\nPrimeiras linhas de df_raw:")
    print(df_raw.head())
    print("\nColunas de df_raw:", list(df_raw.columns))

    if len(df_raw) > max_linhas_final:
        df_raw = df_raw.sample(n=max_linhas_final, random_state=42).reset_index(drop=True)
        print(f"\n>>> Amostra reduzida para {df_raw.shape} linhas para facilitar análise.")

    return df_raw


def preparar_dataframe(df_raw: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
    # Aqui meus caros vamos mudar os nomes das colunas
    rename_map = {
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
        "UMIDADE RELATIVA DO AR, \nHORARIA (%)": "umid_pct",
        "UMIDADE RELATIVA DO AR, HORARIA (%)": "umid_pct",
        "VENTO, DIRECAO HORARIA (gr)": "vento_dir_graus",
        "VENTO, RAJADA MAXIMA (m/s)": "vento_rajada_ms",
        "VENTO, VELOCIDADE HORARIA (m/s)": "vento_vel_ms",
    }
    df_raw = df_raw.rename(columns=rename_map)

    # Converter tipos
    df_raw["data"] = pd.to_datetime(df_raw["data"], errors="coerce")
    df_raw["hora_utc"] = pd.to_numeric(df_raw["hora_utc"], errors="coerce")

    # Definir lista de colunas numéricas climáticas. Pesquisar depois os significados
    colunas_numericas = [
        "precipitacao_mm", "pressao_estacao_mb", "pressao_max_mb", "pressao_min_mb",
        "radiacao_wm2", "temp_bulbo_seco_c", "temp_orvalho_c", "temp_max_c", "temp_min_c",
        "temp_orvalho_max_c", "temp_orvalho_min_c", "umid_max_pct", "umid_min_pct", "umid_pct",
        "vento_dir_graus", "vento_rajada_ms", "vento_vel_ms"
    ]
    for col in colunas_numericas:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            # TRATAMENTO CRÍTICO: remover sentinelas -9999
            df_raw.loc[df_raw[col] <= -9000, col] = np.nan

    
    df_raw = df_raw.dropna(subset=["data"]).copy()

    # Partes da data
    df_raw["ano"] = df_raw["data"].dt.year
    df_raw["mes"] = df_raw["data"].dt.month
    df_raw["dia"] = df_raw["data"].dt.day
    df_raw["estacao_ano"] = df_raw["mes"].apply(definir_estacao_ano_br)

    # Hora 0–23 para facilitar no BI
    df_raw["hora"] = (df_raw["hora_utc"] // 100).astype("Int64")

    print("\nExemplo de colunas de data e estacao_ano:")
    print(df_raw[["data", "ano", "mes", "dia", "hora_utc", "hora", "estacao_ano"]].head())

    
    df = df_raw.merge(
        df_stations[["estacao", "uf", "regiao", "latitude", "longitude", "altitude_m"]],
        on="estacao",
        how="left",
    )

    print("\n>>> df (clima + info estação) após merge:")
    print("Formato:", df.shape)
    print(df[["estacao", "uf", "regiao"]].head())

    # Codificar categóricas
    for col_cat in ["uf", "regiao", "estacao_ano"]:
        df[f"{col_cat}_cod"] = df[col_cat].astype("category").cat.codes

    # Variável alvo choveu
    # (depois do tratamento de NaN: se precipitação não foi medida, assume 0)
    if "precipitacao_mm" in df.columns:
        df["precipitacao_mm"] = df["precipitacao_mm"].fillna(0)
    df["choveu"] = (df["precipitacao_mm"] > 0).astype(int)

    print("\nDistribuição de choveu (0=sem chuva, 1=com chuva):")
    print(df["choveu"].value_counts())
    print("\nProporções de choveu:")
    print(df["choveu"].value_counts(normalize=True).rename("proportion"))

    
    cols_para_preencher = [
        "pressao_estacao_mb", "pressao_max_mb", "pressao_min_mb", "radiacao_wm2",
        "temp_bulbo_seco_c", "temp_orvalho_c", "temp_max_c", "temp_min_c",
        "temp_orvalho_max_c", "temp_orvalho_min_c", "umid_max_pct", "umid_min_pct",
        "umid_pct", "vento_dir_graus", "vento_rajada_ms", "vento_vel_ms",
        "altitude_m", "latitude", "longitude", "hora_utc"
    ]
    cols_exist = [c for c in cols_para_preencher if c in df.columns]
    df[cols_exist] = df[cols_exist].fillna(df[cols_exist].median())

    return df


def analise_univariada(df: pd.DataFrame, plots_dir: str, outputs_dir: str):
    print("\n=== ANÁLISE UNIVARIADA ===")
    variaveis = [
        "temp_bulbo_seco_c", "temp_max_c", "temp_min_c", "temp_orvalho_c",
        "umid_pct", "umid_max_pct", "umid_min_pct",
        "pressao_estacao_mb", "radiacao_wm2",
        "vento_vel_ms", "altitude_m",
    ]
    print("Variáveis selecionadas para univariada:", variaveis)

    linhas = []
    for col in variaveis:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        if serie.empty:
            continue

        media = serie.mean()
        mediana = serie.median()
        moda = serie.mode().iloc[0] if not serie.mode().empty else np.nan
        desvio = serie.std()
        p25 = serie.quantile(0.25)
        p50 = serie.quantile(0.50)
        p75 = serie.quantile(0.75)

        linhas.append(
            dict(variavel=col, media=media, mediana=mediana, moda=moda,
                 desvio_padrao=desvio, p25=p25, p50=p50, p75=p75)
        )

        plt.figure(figsize=(6, 4))
        sns.histplot(serie, bins=30, kde=False)
        plt.title(f"Histograma - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()
        caminho_hist = os.path.join(plots_dir, f"hist_{col}.png")
        plt.savefig(caminho_hist)
        plt.close()
        print(f"Histograma salvo: {caminho_hist}")

    df_uni = pd.DataFrame(linhas)
    caminho_resumo = os.path.join(outputs_dir, "univariada_resumo.csv")
    df_uni.to_csv(caminho_resumo, index=False, float_format="%.6f")
    print("\nResumo univariado salvo em:", caminho_resumo)
    print(df_uni)


def analise_multivariada(df: pd.DataFrame, plots_dir: str, outputs_dir: str):
    print("\n=== ANÁLISE MULTIVARIADA (CORRELAÇÃO DE PEARSON) ===")
    cols_corr = [
        "precipitacao_mm", "temp_bulbo_seco_c", "temp_max_c", "temp_min_c",
        "temp_orvalho_c", "temp_orvalho_max_c", "temp_orvalho_min_c",
        "umid_pct", "umid_max_pct", "umid_min_pct",
        "pressao_estacao_mb", "pressao_max_mb", "pressao_min_mb",
        "radiacao_wm2", "vento_vel_ms", "vento_rajada_ms",
        "altitude_m", "latitude", "longitude",
        "ano", "mes", "dia", "hora_utc",
        "uf_cod", "regiao_cod", "estacao_ano_cod",
        "choveu",
    ]
    cols_corr = [c for c in cols_corr if c in df.columns]
    corr = df[cols_corr].corr(method="pearson")

    caminho_corr = os.path.join(outputs_dir, "correlacao_pearson.csv")
    corr.to_csv(caminho_corr, float_format="%.6f")
    print("Matriz de correlação salva em:", caminho_corr)

    print("\nCorrelação de Pearson com a variável alvo 'choveu':")
    print(corr["choveu"].sort_values(ascending=False))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False)
    plt.title("Matriz de correlação de Pearson")
    plt.tight_layout()
    caminho_heatmap = os.path.join(plots_dir, "correlacao_pearson_heatmap.png")
    plt.savefig(caminho_heatmap)
    plt.close()
    print("Heatmap de correlação salvo em:", caminho_heatmap)


def visualizacoes(df: pd.DataFrame, plots_dir: str):
    print("\n=== VISUALIZAÇÕES ===")

    # 1) Boxplot de precipitação por estação do ano (apenas registros com chuva) vai ser dificil colocar no pwer BI
    df_precip = df[df["precipitacao_mm"] > 0].copy()
    if not df_precip.empty:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="estacao_ano",
            y="precipitacao_mm",
            data=df_precip,
            order=["verao", "outono", "inverno", "primavera"],
        )
        plt.title("Precipitação (mm) por estação do ano (registros com chuva)")
        plt.xlabel("Estação do ano")
        plt.ylabel("Precipitação (mm)")
        plt.tight_layout()
        caminho_box = os.path.join(plots_dir, "boxplot_precipitacao_por_estacao_ano.png")
        plt.savefig(caminho_box)
        plt.close()
        print("Gráfico salvo:", caminho_box)

    # proporção de registros com chuva por estação do ano
    prop_chuva_estacao = (
        df.groupby("estacao_ano")["choveu"]
        .mean()
        .reindex(["verao", "outono", "inverno", "primavera"])
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(x=prop_chuva_estacao.index, y=prop_chuva_estacao.values)
    plt.title("Proporção de registros com chuva por estação do ano")
    plt.xlabel("Estação do ano")
    plt.ylabel("Proporção de registros com chuva")
    plt.tight_layout()
    caminho_bar = os.path.join(plots_dir, "barra_choveu_por_estacao_ano.png")
    plt.savefig(caminho_bar)
    plt.close()
    print("Gráfico salvo:", caminho_bar)

    #  temperatura média por hora do dia
    temp_hora = df.groupby("hora")["temp_bulbo_seco_c"].mean().sort_index()
    plt.figure(figsize=(8, 4))
    temp_hora.plot(kind="line")
    plt.title("Temperatura média do ar por hora do dia (UTC)")
    plt.xlabel("Hora do dia (0–23)")
    plt.ylabel("Temperatura média (°C)")
    plt.tight_layout()
    caminho_linha = os.path.join(plots_dir, "linha_temp_media_por_hora.png")
    plt.savefig(caminho_linha)
    plt.close()
    print("Gráfico salvo:", caminho_linha)


def modelos_classificacao(df: pd.DataFrame, outputs_dir: str):
    print("\n=== MODELOS DE CLASSIFICAÇÃO (ML) ===")

    features = [
        "temp_bulbo_seco_c", "temp_max_c", "temp_min_c", "temp_orvalho_c",
        "umid_pct", "umid_max_pct", "umid_min_pct",
        "pressao_estacao_mb", "radiacao_wm2",
        "vento_vel_ms", "vento_rajada_ms",
        "hora_utc", "mes", "ano",
        "altitude_m", "latitude", "longitude",
        "uf_cod", "regiao_cod", "estacao_ano_cod",
    ]
    features = [f for f in features if f in df.columns]
    print("Features usadas no ML:", features)

    # Amostra para ML se necessário
    max_linhas_ml = 120_000
    if len(df) > max_linhas_ml:
        df_ml = df.sample(n=max_linhas_ml, random_state=42)
    else:
        df_ml = df.copy()

    X = df_ml[features].copy()
    y = df_ml["choveu"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = []

    # 1) Logistic Regression Critério exigido pelo professor
    print("\n--- Treinando modelo: Logistic Regression ---")
    logreg = LogisticRegression(max_iter=1500, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    y_pred_log = logreg.predict(X_test_scaled)
    acc_log = accuracy_score(y_test, y_pred_log)
    print("Acurácia:", round(acc_log, 4))
    print("Classification report:")
    print(classification_report(y_test, y_pred_log))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred_log))
    resultados.append({"modelo": "Logistic Regression", "acuracia": acc_log})

    # 2) Random Forest 
    print("\n--- Treinando modelo: Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Acurácia:", round(acc_rf, 4))
    print("Classification report:")
    print(classification_report(y_test, y_pred_rf))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred_rf))
    resultados.append({"modelo": "Random Forest", "acuracia": acc_rf})

    # 3) KNN
    print("\n--- Treinando modelo: KNN ---")
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print("Acurácia:", round(acc_knn, 4))
    print("Classification report:")
    print(classification_report(y_test, y_pred_knn))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred_knn))
    resultados.append({"modelo": "KNN", "acuracia": acc_knn})

    df_res = pd.DataFrame(resultados)
    caminho_res = os.path.join(outputs_dir, "resultados_modelos_ml.csv")
    df_res.to_csv(caminho_res, index=False, float_format="%.6f")
    print("\nResumo dos modelos salvo em:", caminho_res)
    print(df_res)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(">>> Iniciando análise do clima (brazilWeather)")
    data_dir = os.path.join(base_dir, "data")
    print("Caminho da pasta de dados:", data_dir)

    outputs_dir, plots_dir = garantir_pastas(base_dir)

 
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    print(">>> Arquivos encontrados:")
    for f in csv_files:
        print(" -", os.path.join("./data", f))
    print("Quantidade de arquivos CSV encontrados:", len(csv_files))

    # Identificar arquivos
    stations_file = None
    dados_file = None
    for f in csv_files:
        nome_lower = f.lower()
        if "codes" in nome_lower or "codigo" in nome_lower:
            stations_file = os.path.join(data_dir, f)
        elif "weather" in nome_lower:
            dados_file = os.path.join(data_dir, f)
    if stations_file is None or dados_file is None:
        raise FileNotFoundError("Não encontrei os arquivos de estações e/ou dados climáticos em ./data.")

    df_stations = carregar_estacoes(stations_file)
    df_raw = carregar_dados_clima_amostrado(dados_file, frac_por_chunk=0.02, max_linhas_final=150000)
    df = preparar_dataframe(df_raw, df_stations)

    # Export para Power BI
    caminho_powerbi = os.path.join(outputs_dir, "df_amostra_trabalho.csv")
    df.to_csv(caminho_powerbi, index=False)
    print("\nArquivo para Power BI salvo em:", caminho_powerbi)

    # Análises
    analise_univariada(df, plots_dir, outputs_dir)
    analise_multivariada(df, plots_dir, outputs_dir)
    visualizacoes(df, plots_dir)
    modelos_classificacao(df, outputs_dir)

    print("\n>>> Fim da execução do analise_clima.py")


if __name__ == "__main__":
    main()
