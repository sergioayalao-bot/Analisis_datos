# ══════════════════════════════════════════════════════════════
# CELDA 1 — Preparación del dataset y análisis bivariado completo
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ── Instalar librerías adicionales ────────────────────────────
!pip install shap -q

# ── Estilo visual ─────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 130, 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
AZUL  = '#1F4E79'
ROJO  = '#C00000'
GRIS  = '#A9A9A9'

# ── Carga del dataset ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

RUTA = '/content/drive/MyDrive/proyecto2/aca2data_limpia.xlsx'
df   = pd.read_excel(RUTA, dtype=str)

# ── Conversiones de tipo ──────────────────────────────────────
df['edad']             = pd.to_numeric(df['edad'], errors='coerce')
df['ano_notificacion'] = pd.to_numeric(df['ano_notificacion'], errors='coerce')

COLS_BIN = ['enfermedades_dolorosas','maltrato_sexual','muerte_familiar',
            'conflicto_pareja','problemas_economicos','esc_educ',
            'problemas_juridicos','problemas_laborales','suicidio_amigo']

for col in COLS_BIN:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in ['localidad_residencia','nombre_upz','ciclovital',
            'clasificaciondelaconducta','sexo','niveleducativo',
            'poblacion_diferencial']:
    df[col] = df[col].str.strip().str.title()

# ── Variable dependiente binaria ──────────────────────────────
# 0 = Ideación suicida  |  1 = Intento de suicidio
df['conducta_bin'] = df['clasificaciondelaconducta'].map({
    'Ideación Suicida'    : 0,
    'Intento De Suicidio' : 1
})
df = df.dropna(subset=['conducta_bin'])
df['conducta_bin'] = df['conducta_bin'].astype(int)

ETIQUETAS = {
    'enfermedades_dolorosas': 'Enfermedades dolorosas',
    'maltrato_sexual'       : 'Maltrato sexual',
    'muerte_familiar'       : 'Muerte familiar',
    'conflicto_pareja'      : 'Conflicto de pareja',
    'problemas_economicos'  : 'Problemas económicos',
    'esc_educ'              : 'Escenario educativo',
    'problemas_juridicos'   : 'Problemas jurídicos',
    'problemas_laborales'   : 'Problemas laborales',
    'suicidio_amigo'        : 'Suicidio de amigo/a'
}

print(f"✅ Dataset listo — {len(df):,} registros válidos")
print(f"   Ideación (0) : {(df['conducta_bin']==0).sum():,}")
print(f"   Intento  (1) : {(df['conducta_bin']==1).sum():,}")

# ── Función V de Cramér ───────────────────────────────────────
def cramer_v(tabla):
    chi2, _, _, _ = chi2_contingency(tabla)
    n = tabla.sum().sum()
    k = min(tabla.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0

# ── Análisis bivariado ────────────────────────────────────────
resultados = []

for col in COLS_BIN:
    sub   = df[[col, 'conducta_bin']].dropna()
    tabla = pd.crosstab(sub[col], sub['conducta_bin'])
    if tabla.shape != (2, 2):
        continue

    chi2_val, p_val, _, _ = chi2_contingency(tabla)
    v        = cramer_v(tabla)
    prev_si  = tabla.loc[1,1] / tabla.loc[1].sum() * 100
    prev_no  = tabla.loc[0,1] / tabla.loc[0].sum() * 100

    try:
        mod    = smf.glm(f'conducta_bin ~ {col}', data=sub,
                         family=sm.families.Poisson(sm.families.links.Log())
                         ).fit(cov_type='HC3')
        rp     = np.exp(mod.params[col])
        rp_low = np.exp(mod.conf_int().loc[col, 0])
        rp_hi  = np.exp(mod.conf_int().loc[col, 1])
        p_rp   = mod.pvalues[col]
    except:
        rp = rp_low = rp_hi = p_rp = np.nan

    resultados.append({
        'Factor'             : ETIQUETAS[col],
        'Prev. con factor %' : round(prev_si, 1),
        'Prev. sin factor %' : round(prev_no, 1),
        'V Cramér'           : round(v, 3),
        'Fuerza'             : 'Fuerte' if v>=.30 else 'Moderada' if v>=.10 else 'Débil',
        'RP'                 : round(rp, 3),
        'IC inf 95%'         : round(rp_low, 3),
        'IC sup 95%'         : round(rp_hi, 3),
        'p-valor'            : round(p_rp, 4),
        'Sig.'               : '✅' if p_rp < 0.05 else '—'
    })

df_biv = pd.DataFrame(resultados).sort_values('RP', ascending=False)

print("\nTABLA BIVARIADA — Factores de riesgo vs. Intento de suicidio")
print("="*85)
print(df_biv.to_string(index=False))

# ── Forest Plot ───────────────────────────────────────────────
df_fp = df_biv.dropna(subset=['RP']).sort_values('RP').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))

for i, row in df_fp.iterrows():
    color = ROJO if row['Sig.'] == '✅' else GRIS
    ax.plot([row['IC inf 95%'], row['IC sup 95%']], [i, i],
            color=color, linewidth=2.5, solid_capstyle='round', zorder=2)
    ax.scatter(row['RP'], i, color=color, s=90, zorder=3)
    ax.text(df_fp['IC sup 95%'].max() * 1.02, i,
            f"RP={row['RP']:.2f}  [{row['IC inf 95%']:.2f}–{row['IC sup 95%']:.2f}]  "
            f"p={row['p-valor']:.3f}",
            va='center', fontsize=8, color=AZUL)

ax.axvline(1.0, color='black', linestyle='--', linewidth=1.3)
ax.set_yticks(range(len(df_fp)))
ax.set_yticklabels(df_fp['Factor'], fontsize=9)
ax.set_xlabel('Razón de Prevalencia  |  IC 95%')
ax.set_title('Forest Plot — Factores de riesgo asociados al intento de suicidio\n'
             'Bogotá D.C., 2023–2025', pad=12)
ax.legend(handles=[
    mpatches.Patch(color=ROJO, label='Significativo  p < 0.05'),
    mpatches.Patch(color=GRIS, label='No significativo')
], fontsize=9, loc='lower right')

sns.despine()
plt.tight_layout()
plt.savefig('C1_forest_RP.png', bbox_inches='tight')
plt.show()
print("✅ Celda 1 completada")



import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Configuración de variables
vars_modelo = COLS_BIN + ['edad']
labels_map = {**ETIQUETAS, 'edad': 'Edad'}

df_m = df[vars_modelo + ['conducta_bin']].dropna().copy()

# --- Multicolinealidad (VIF) ---
X_vif = sm.add_constant(df_m[vars_modelo])
vif_vals = [variance_inflation_factor(X_vif.values, i + 1) for i in range(len(vars_modelo))]

vif = pd.DataFrame({
    'Variable': [labels_map.get(v, v) for v in vars_modelo],
    'VIF': vif_vals
})

def eval_vif(val):
    if val < 5: return 'Aceptable'
    if val < 10: return 'Moderado'
    return 'Alto'

vif['Estado'] = vif['VIF'].apply(eval_vif)

print("VIF - Multicolinealidad:")
print(vif.sort_values('VIF', ascending=False).to_string(index=False))

# --- Regresión Logística ---
X = sm.add_constant(df_m[vars_modelo])
y = df_m['conducta_bin']

model = sm.Logit(y, X).fit(maxiter=300, disp=False)

# Métricas del modelo y R2 Nagelkerke
ll_null, ll_fit, n = model.llnull, model.llf, len(y)
r2_nagelkerke = (1 - np.exp(2 * (ll_null - ll_fit) / n)) / (1 - np.exp(2 * ll_null / n))

coef = model.params.iloc[1:]
ci = model.conf_int().iloc[1:]
pvals = model.pvalues.iloc[1:]

tabla_or = pd.DataFrame({
    'Variable': [labels_map.get(x, x) for x in coef.index],
    'OR': np.exp(coef),
    'IC_inf': np.exp(ci[0]),
    'IC_sup': np.exp(ci[1]),
    'p_value': pvals,
    'Sig': pvals < 0.05
}).sort_values('OR', ascending=False).reset_index(drop=True)

# Impresión de resultados
print("\nResults - Odds Ratios Ajustados:")
print(tabla_or.round(4).to_string(index=False))
print(f"\nR2 Nagelkerke: {r2_nagelkerke:.4f} | AIC: {model.aic:.2f} | BIC: {model.bic:.2f}\n")

# --- Visualización (Forest Plot) ---
df_plot = tabla_or.sort_values('OR').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5))

for i, row in df_plot.iterrows():
    c = ROJO if row['Sig'] else GRIS
    ax.plot([row['IC_inf'], row['IC_sup']], [i, i], color=c, lw=2, zorder=2)
    ax.scatter(row['OR'], i, color=c, s=70, zorder=3)
    
    p_txt = '< 0.001' if row['p_value'] < 0.001 else f"{row['p_value']:.3f}"
    lbl = f"OR: {row['OR']:.2f} [{row['IC_inf']:.2f} - {row['IC_sup']:.2f}], p={p_txt}"
    
    ax.text(df_plot['IC_sup'].max() * 1.05, i, lbl, va='center', fontsize=8.5, color=AZUL)

ax.axvline(1.0, color='black', ls='--', lw=1)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['Variable'], fontsize=9)
ax.set_xlabel('Odds Ratio (IC 95%)')
ax.set_title('Forest Plot - Odds Ratios Ajustados (Bogotá D.C.)', loc='left', pad=10)

ax.legend(handles=[
    mpatches.Patch(color=ROJO, label='p < 0.05'),
    mpatches.Patch(color=GRIS, label='n.s.')
], loc='lower right', frameon=True)

sns.despine()
plt.tight_layout()
plt.savefig('C2_forest_OR.png', dpi=300, bbox_inches='tight')
plt.show()

#------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

palette = ['#1F4E79', '#C00000', '#2E75B6', '#FF8C00', '#4CAF50', '#9C27B0']

# --- Prep de datos ---
vars_clust = COLS_BIN + ['edad']
meta_cols = ['conducta_bin', 'sexo', 'ciclovital', 'localidad_residencia']

df_cl = df[vars_clust + meta_cols].dropna().copy()

X_cl = df_cl[vars_clust].copy()
X_cl['edad'] = StandardScaler().fit_transform(X_cl[['edad']])
X_mat = X_cl.values

# Muestra para validación rápida
np.random.seed(42)
idx_eval = np.random.choice(len(X_mat), size=min(30_000, len(X_mat)), replace=False)
X_eval = X_mat[idx_eval]

# --- Evaluación de k (Codo y Silueta) ---
k_range = range(2, 9)
inertias, silhouettes = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_eval)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_eval, labels)
    silhouettes.append(sil)
    print(f"k={k} | Inertia: {km.inertia_:12,.0f} | Silhouette: {sil:.4f}")

k_opt = k_range[np.argmax(silhouettes)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Codo
ax1.plot(list(k_range), inertias, 'o-', color=AZUL, lw=2, ms=6)
ax1.axvline(k_opt, color=ROJO, ls='--', lw=1.2, label=f'k sugerido = {k_opt}')
ax1.set(xlabel='Número de clústeres (k)', ylabel='Inercia', title='Método del Codo')
ax1.set_xticks(list(k_range))
ax1.legend(frameon=True)
sns.despine(ax=ax1)

# Silueta
bar_colors = [ROJO if k == k_opt else '#2E75B6' for k in k_range]
ax2.bar(list(k_range), silhouettes, color=bar_colors, width=0.6)
for k, s in zip(k_range, silhouettes):
    ax2.text(k, s + 0.002, f'{s:.3f}', ha='center', fontsize=8, color=AZUL)

ax2.set(xlabel='Número de clústeres (k)', ylabel='Silueta Promedio', title='Índice de Silueta')
ax2.set_xticks(list(k_range))
sns.despine(ax=ax2)

plt.suptitle('Evaluación de k - Bogotá D.C. (2023-2025)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('D1_codo_silueta.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Clustering Final ---
k_final = k_opt
km_final = KMeans(n_clusters=k_final, random_state=42, n_init=15, max_iter=500)
df_cl['cluster'] = km_final.fit_predict(X_mat)

print(f"\nModel fit (k={k_final}) | Silhouette: {silhouette_score(X_eval, km_final.predict(X_eval)):.4f}\n")

# Resumen de clústeres
summary = df_cl.groupby('cluster').agg(
    n=('edad', 'count'),
    pct=('edad', lambda x: len(x) / len(df_cl) * 100),
    edad_mean=('edad', 'mean'),
    pct_intento=('conducta_bin', lambda x: x.mean() * 100)
)
print(summary.round(2).to_string())

# --- Heatmap de Perfiles ---
profile = df_cl.groupby('cluster')[COLS_BIN].mean().mul(100).round(1)
profile.index = [f'Clúster {i}' for i in profile.index]
profile.columns = [ETIQUETAS.get(c, c) for c in COLS_BIN]

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    profile.T, annot=True, fmt='.1f', cmap='YlOrRd',
    lw=0.5, annot_kws={'size': 9},
    cbar_kws={'label': '% de casos con factor presente'}, ax=ax
)
ax.set_title('Prevalencia de factores de riesgo por clúster (%)', loc='left', pad=12)
plt.tight_layout()
plt.savefig('F1_heatmap_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Proyección PCA 2D ---
n_pca = min(50_000, len(X_mat))
idx_pca = np.random.choice(len(X_mat), size=n_pca, replace=False)

X_pca_sub = X_mat[idx_pca]
labs_sub = df_cl['cluster'].values[idx_pca]
cond_sub = df_cl['conducta_bin'].values[idx_pca]

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_pca_sub)
ve = pca.explained_variance_ratio_

print(f"\nPCA Exp. Variance: PC1={ve[0]:.1%}, PC2={ve[1]:.1%} (Total: {sum(ve):.1%})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Clústeres
for cl in range(k_final):
    mask = labs_sub == cl
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], s=2, alpha=0.2,
                color=palette[cl % len(palette)], label=f'Clúster {cl}')

centroids_2d = pca.transform(km_final.cluster_centers_)
ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=120, marker='X', color='black', zorder=5, label='Centroides')
ax1.set(xlabel=f'PC1 ({ve[0]:.1%})', ylabel=f'PC2 ({ve[1]:.1%})', title='Clústeres en Espacio PCA')
ax1.legend(markerscale=3, fontsize=8, loc='upper right')
sns.despine(ax=ax1)

# Plot 2: Conducta
cond_colors = np.where(cond_sub == 0, AZUL, ROJO)
ax2.scatter(X_2d[:, 0], X_2d[:, 1], s=2, alpha=0.15, c=cond_colors)
ax2.set(xlabel=f'PC1 ({ve[0]:.1%})', ylabel=f'PC2 ({ve[1]:.1%})', title='Conducta en Espacio PCA')
ax2.legend(handles=[
    mpatches.Patch(color=AZUL, label='Ideación'),
    mpatches.Patch(color=ROJO, label='Intento')
], loc='upper right')
sns.despine(ax=ax2)

plt.suptitle('Proyección PCA 2D', fontsize=12)
plt.tight_layout()
plt.savefig('G1_PCA_clusters.png', dpi=300, bbox_inches='tight')
plt.show()


#--------------
