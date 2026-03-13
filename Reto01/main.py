import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

_df = load_penguins().dropna().reset_index(drop=True)
_X = _df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
_y = _df['species']

_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

_modelo_ml = DecisionTreeClassifier(random_state=42)
_modelo_ml.fit(_X_train, _y_train)


def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Clasifica un pinguino basandose en reglas diseñadas manualmente.

    Estrategia:
    1. Gentoo: aleta >= 207mm, o pico delgado (bd < 15.9) combinado con aleta >= 203mm.
       El doble criterio evita capturar Adelie con bill_depth bajo y aleta corta.
    2. Chinstrap con bill_depth >= 20 siempre tiene bill_length >= 50; Adelie no.
    3. bill_length < 37 es siempre Adelie (Chinstrap minimo es ~41mm).
    4. bill_length >= 46 es siempre Chinstrap.
    5. Zona 37-41: Adelie si bill_depth >= 16.5.
    6. Zona gris 41-46 con bd >= 18.5: Adelie salvo que bl >= 45 (Chinstrap largo).
    7. Zona gris 41-46 con bd 17-18.5: usar body_mass >= 3900 como desempate.

    Retorna: 'Adelie', 'Chinstrap' o 'Gentoo'
    """
    if flipper_length_mm >= 207:
        return 'Gentoo'

    if bill_depth_mm < 15.9 and flipper_length_mm >= 203:
        return 'Gentoo'

    if bill_depth_mm >= 20.0:
        if bill_length_mm >= 50:
            return 'Chinstrap'
        return 'Adelie'

    if bill_length_mm < 37:
        return 'Adelie'

    if bill_length_mm >= 46:
        return 'Chinstrap'

    if bill_length_mm <= 41:
        if bill_depth_mm >= 16.5:
            return 'Adelie'
        return 'Chinstrap'

    if bill_depth_mm >= 18.5:
        if bill_length_mm >= 45.0:
            return 'Chinstrap'
        return 'Adelie'

    if bill_depth_mm >= 17.0:
        if body_mass_g >= 3900:
            return 'Adelie'
        return 'Chinstrap'

    return 'Chinstrap'


def clasificador_ml(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Clasifica un pinguino usando un DecisionTreeClassifier entrenado con
    sklearn sobre el dataset Palmer Penguins (train/test 80/20, random_state=42).

    Retorna: 'Adelie', 'Chinstrap' o 'Gentoo'
    """
    features = pd.DataFrame([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g]],
                            columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    return _modelo_ml.predict(features)[0]


if __name__ == '__main__':
    import sys
    import os
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df_input = pd.read_csv(csv_path).dropna(subset=FEATURES).reset_index(drop=True)
    else:
        df_input = _df.copy()
        print("Uso: python main.py <archivo.csv>")
        print("No se proporcionó CSV, usando el dataset interno de Palmer Penguins.\n")

    preds_humano = [
        clasificador_humano(r.bill_length_mm, r.bill_depth_mm, r.flipper_length_mm, r.body_mass_g)
        for _, r in df_input.iterrows()
    ]
    preds_ml = [
        clasificador_ml(r.bill_length_mm, r.bill_depth_mm, r.flipper_length_mm, r.body_mass_g)
        for _, r in df_input.iterrows()
    ]

    df_humano = df_input.copy()
    df_humano['species'] = preds_humano

    df_ml = df_input.copy()
    df_ml['species'] = preds_ml

    # Asegurar que species sea la primera columna, igual que el dataset original
    other_cols = [c for c in df_humano.columns if c != 'species']
    df_humano = df_humano[['species'] + other_cols]
    df_ml = df_ml[['species'] + other_cols]

    base = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else 'penguins'
    out_humano = f'resultados_humano_{base}.csv'
    out_ml = f'resultados_ml_{base}.csv'

    df_humano.to_csv(out_humano, index=False)
    df_ml.to_csv(out_ml, index=False)

    print(f"Resultados guardados en:")
    print(f"  {out_humano}  ({len(df_humano)} filas)")
    print(f"  {out_ml}  ({len(df_ml)} filas)")

    has_labels = 'species' in pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else '').columns if len(sys.argv) > 1 else True

    if has_labels or len(sys.argv) == 1:
        y_true = _df['species'].tolist() if len(sys.argv) == 1 else pd.read_csv(sys.argv[1]).dropna(subset=FEATURES)['species'].tolist()

        def metricas(y_true, y_pred):
            return {
                'Accuracy':  accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'Aciertos':  sum(p == r for p, r in zip(y_pred, y_true)),
                'Errores':   sum(p != r for p, r in zip(y_pred, y_true)),
            }

        m_h = metricas(y_true, preds_humano)
        m_ml = metricas(y_true, preds_ml)
        total = len(y_true)
        ganador = 'Humano' if m_h['Accuracy'] > m_ml['Accuracy'] else ('ML' if m_ml['Accuracy'] > m_h['Accuracy'] else 'Empate')

        print()
        sep = '+' + '-'*22 + '+' + '-'*16 + '+' + '-'*16 + '+'
        print(sep)
        print(f"| {'Metrica':<20} | {'Humano':>14} | {'ML (DecisionTree)':>14} |")
        print(sep)
        for metrica in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            print(f"| {metrica:<20} | {m_h[metrica]:>14.4f} | {m_ml[metrica]:>14.4f} |")
        print(sep)
        ac_h = f"{m_h['Aciertos']}/{total}"
        ac_ml = f"{m_ml['Aciertos']}/{total}"
        print(f"| {'Aciertos':<20} | {ac_h:>14} | {ac_ml:>14} |")
        print(f"| {'Errores':<20} | {m_h['Errores']:>14} | {m_ml['Errores']:>14} |")
        print(sep)
        print(f"| {'Ganador':<20} | {ganador:>30} |")
        print(sep)
