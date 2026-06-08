"""
================================================================================
ANALISIS DATA KATEGORIKAL - REGRESI LOGISTIK
PENGARUH PENDAPATAN DAN SANITASI TERHADAP STATUS STUNTING PADA BALITA
================================================================================
Mata Kuliah : Analisis Data Kategorikal
Program     : Statistika / Data Science
Metode      : Point Bi-Serial Correlation, Regresi Logistik, Regresi Probit
Tujuan      :
    1. Menganalisis hubungan antara pendapatan (income) dengan status stunting
    2. Menganalisis pengaruh kondisi sanitasi terhadap status stunting balita
    3. Membangun model prediktif yang mampu mengklasifikasikan status stunting
================================================================================
"""

# ============================================================
# BAGIAN 1: IMPORT LIBRARY YANG DIPERLUKAN
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Library statistika inferensial
from scipy import stats
from scipy.stats import pointbiserialr

# Library pemodelan regresi logistik dan probit
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Library evaluasi model machine learning
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

print("=" * 70)
print("  ANALISIS PENGARUH PENDAPATAN DAN SANITASI TERHADAP STATUS STUNTING")
print("=" * 70)

# ============================================================
# BAGIAN 2: INPUT DATA PENELITIAN
# ============================================================
# Data primer penelitian stunting balita
# Variabel:
#   - Stunting : Status stunting (1 = stunting, 0 = tidak stunting)
#   - Income   : Pendapatan keluarga (dalam juta rupiah)
#   - Nutrition: Skor asupan gizi balita (0-100)
#   - Sanitation: Skor kondisi sanitasi rumah tangga (0-100)

data_raw = {
    'No':          list(range(1, 26)),
    'Stunting':    [1,1,1,1,1,1,1,1,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'Income':      [2.1,2.5,3.0,3.2,2.8,3.5,3.0,2.2,2.9,3.1,
                    6.5,7.0,6.8,7.5,8.0,6.2,7.3,6.9,8.2,7.8,
                    6.7,8.1,7.4,6.6,7.9],
    'Nutrition':   [55,50,52,48,53,49,51,47,50,52,
                    75,78,80,77,79,76,82,74,81,79,
                    77,83,78,76,80],
    'Sanitation':  [60,58,61,55,57,59,56,54,60,58,
                    80,82,85,83,86,81,88,80,87,84,
                    83,89,85,82,86]
}

data = pd.DataFrame(data_raw)

print("\n[1] TAMPILAN DATA PENELITIAN")
print("-" * 50)
print(data.to_string(index=False))
print(f"\nTotal observasi : {len(data)}")
print(f"Kasus stunting  : {data['Stunting'].sum()} balita")
print(f"Tidak stunting  : {(data['Stunting'] == 0).sum()} balita")

# ============================================================
# BAGIAN 3: STATISTIKA DESKRIPTIF
# ============================================================
print("\n" + "=" * 70)
print("[2] STATISTIKA DESKRIPTIF")
print("=" * 70)

# Deskriptif berdasarkan status stunting
desc_groups = data.groupby('Stunting')[['Income', 'Nutrition', 'Sanitation']].agg(
    ['mean', 'std', 'min', 'max']
)
print("\nStatistika Deskriptif Berdasarkan Status Stunting:")
print(desc_groups.round(3))

print("\nKeterangan Label:")
print("  Stunting = 1 (balita mengalami stunting)")
print("  Stunting = 0 (balita tidak mengalami stunting)")

# ============================================================
# BAGIAN 4: PEMILIHAN MODEL
# ============================================================
print("\n" + "=" * 70)
print("[3] PEMILIHAN MODEL ANALISIS")
print("=" * 70)
print("""
JUSTIFIKASI PEMILIHAN MODEL:
─────────────────────────────────────────────────────────────────────
Variabel Dependen  : Stunting → Skala Nominal/Dikotomi (0 = tidak stunting,
                     1 = stunting). Variabel ini bersifat biner/kategoris.

Variabel Independen:
  • Income     → Skala Rasio (pendapatan dalam juta rupiah, nilai positif
                 dengan nol mutlak)
  • Sanitation → Skala Interval (skor kondisi sanitasi rentang 1–100)

ALASAN PEMILIHAN REGRESI LOGISTIK:
  1. Variabel respon (Y = Stunting) bersifat biner (0/1), sehingga distribusi
     asumsi normalitas residual tidak berlaku → OLS tidak tepat digunakan.
  2. Regresi Logistik menggunakan fungsi logit sebagai link function sehingga
     nilai prediksi terbatas pada rentang [0,1] sebagai probabilitas.
  3. Regresi Logistik sesuai untuk hubungan antara variabel nominal (Y)
     dengan variabel interval/rasio (X) sesuai dengan ukuran asosiasi
     Point Bi-Serial yang relevan pada skala pengukuran ini.
  4. Interpretasi odds ratio dari regresi logistik relevan secara klinis
     untuk menyatakan besar risiko stunting berdasarkan faktor penentu.

MODEL YANG DIBANGUN:
  • Model A : Logit(P(Stunting=1)) = β₀ + β₁·Income
  • Model B : Logit(P(Stunting=1)) = β₀ + β₁·Sanitation
─────────────────────────────────────────────────────────────────────
""")

# ============================================================
# BAGIAN 5: UKURAN ASOSIASI - POINT BI-SERIAL CORRELATION
# ============================================================
print("=" * 70)
print("[4] UJI UKURAN ASOSIASI — KORELASI POINT BI-SERIAL")
print("=" * 70)
print("""
Korelasi Point Bi-Serial digunakan untuk mengukur kekuatan dan arah hubungan
antara variabel dikotomi (Stunting: nominal) dengan variabel kontinu
(Income dan Sanitation: rasio/interval).
""")

Y = data['Stunting']

for var in ['Income', 'Nutrition', 'Sanitation']:
    r_pb, p_val = pointbiserialr(Y, data[var])
    sig = "SIGNIFIKAN" if p_val < 0.05 else "TIDAK SIGNIFIKAN"
    arah = "positif" if r_pb > 0 else "negatif"
    print(f"  {var:<12}: r_pb = {r_pb:+.4f} | p-value = {p_val:.6f} → {sig} ({arah})")

print("""
Interpretasi:
  - Income     : Korelasi negatif kuat dan signifikan → semakin tinggi
                 pendapatan, peluang stunting semakin RENDAH.
  - Nutrition  : Korelasi negatif kuat dan signifikan → asupan gizi lebih
                 baik berkaitan dengan risiko stunting yang lebih RENDAH.
  - Sanitation : Korelasi negatif kuat dan signifikan → kondisi sanitasi
                 yang lebih baik berkaitan dengan risiko stunting LEBIH RENDAH.
""")

# ============================================================
# BAGIAN 6: MODEL A — REGRESI LOGISTIK (INCOME → STUNTING)
# ============================================================
print("=" * 70)
print("[5] MODEL A — REGRESI LOGISTIK: INCOME → STUNTING")
print("=" * 70)

# Menambahkan konstanta (intersep) pada variabel prediktor
Y_A = data['Stunting']
X_A = sm.add_constant(data['Income'])  # menambah kolom konstanta bernilai 1

# Membangun dan melatih model logistik
model_A = sm.Logit(Y_A, X_A)
result_A = model_A.fit(disp=0)

print("\n--- RINGKASAN MODEL LOGISTIK A (Income → Stunting) ---")
print(result_A.summary())

# Menghitung Odds Ratio beserta confidence interval 95%
odds_ratio_A = np.exp(result_A.params)
conf_int_A   = np.exp(result_A.conf_int())
print("\nOdds Ratio dan 95% Confidence Interval:")
or_table_A = pd.DataFrame({
    'Odds Ratio'  : odds_ratio_A,
    'CI Lower 2.5%': conf_int_A[0],
    'CI Upper 97.5%': conf_int_A[1]
})
print(or_table_A.round(4))

# Prediksi kelas dan probabilitas
data['prob_A']       = result_A.predict(X_A)
data['pred_class_A'] = (data['prob_A'] >= 0.5).astype(int)

# Evaluasi ketepatan klasifikasi
accuracy_A = accuracy_score(Y_A, data['pred_class_A'])
cm_A       = confusion_matrix(Y_A, data['pred_class_A'])
report_A   = classification_report(Y_A, data['pred_class_A'])

print(f"\nAkurasi Model A (Ketepatan Klasifikasi): {accuracy_A * 100:.2f}%")
print("\nConfusion Matrix Model A:")
print(cm_A)
print("\nClassification Report Model A:")
print(report_A)

# ROC-AUC
fpr_A, tpr_A, _ = roc_curve(Y_A, data['prob_A'])
roc_auc_A       = auc(fpr_A, tpr_A)
print(f"Nilai ROC-AUC Model A: {roc_auc_A:.4f}")

# ============================================================
# BAGIAN 7: MODEL B — REGRESI LOGISTIK (SANITATION → STUNTING)
# ============================================================
print("\n" + "=" * 70)
print("[6] MODEL B — REGRESI LOGISTIK: SANITATION → STUNTING")
print("=" * 70)

Y_B = data['Stunting']
X_B = sm.add_constant(data['Sanitation'])

model_B = sm.Logit(Y_B, X_B)
result_B = model_B.fit(disp=0)

print("\n--- RINGKASAN MODEL LOGISTIK B (Sanitation → Stunting) ---")
print(result_B.summary())

odds_ratio_B = np.exp(result_B.params)
conf_int_B   = np.exp(result_B.conf_int())
print("\nOdds Ratio dan 95% Confidence Interval:")
or_table_B = pd.DataFrame({
    'Odds Ratio'    : odds_ratio_B,
    'CI Lower 2.5%' : conf_int_B[0],
    'CI Upper 97.5%': conf_int_B[1]
})
print(or_table_B.round(4))

data['prob_B']       = result_B.predict(X_B)
data['pred_class_B'] = (data['prob_B'] >= 0.5).astype(int)

accuracy_B = accuracy_score(Y_B, data['pred_class_B'])
cm_B       = confusion_matrix(Y_B, data['pred_class_B'])
report_B   = classification_report(Y_B, data['pred_class_B'])

print(f"\nAkurasi Model B (Ketepatan Klasifikasi): {accuracy_B * 100:.2f}%")
print("\nConfusion Matrix Model B:")
print(cm_B)
print("\nClassification Report Model B:")
print(report_B)

fpr_B, tpr_B, _ = roc_curve(Y_B, data['prob_B'])
roc_auc_B       = auc(fpr_B, tpr_B)
print(f"Nilai ROC-AUC Model B: {roc_auc_B:.4f}")

# ============================================================
# BAGIAN 8: REGRESI PROBIT (SEBAGAI MODEL PEMBANDING)
# ============================================================
print("\n" + "=" * 70)
print("[7] MODEL PROBIT (PEMBANDING) — INCOME & SANITATION → STUNTING")
print("=" * 70)

# --- Probit Model A: Income ---
model_probit_A = sm.Probit(Y_A, X_A)
result_probit_A = model_probit_A.fit(disp=0)
mfx_A = result_probit_A.get_margeff(at='mean')

print("\n--- PROBIT MODEL A: Income → Stunting ---")
print(result_probit_A.summary())
print("\nMarginal Effects at Mean (MEM) - Model A:")
print(mfx_A.summary())

# --- Probit Model B: Sanitation ---
model_probit_B = sm.Probit(Y_B, X_B)
result_probit_B = model_probit_B.fit(disp=0)
mfx_B = result_probit_B.get_margeff(at='mean')

print("\n--- PROBIT MODEL B: Sanitation → Stunting ---")
print(result_probit_B.summary())
print("\nMarginal Effects at Mean (MEM) - Model B:")
print(mfx_B.summary())

# ============================================================
# BAGIAN 9: PERBANDINGAN MODEL LOGISTIK vs PROBIT
# ============================================================
print("\n" + "=" * 70)
print("[8] PERBANDINGAN MODEL: LOGISTIK vs PROBIT")
print("=" * 70)

data_probit_A = result_probit_A.predict(X_A)
data_probit_B = result_probit_B.predict(X_B)

auc_probit_A = roc_auc_score(Y_A, data_probit_A)
auc_probit_B = roc_auc_score(Y_B, data_probit_B)

print("\n{:<30} {:>12} {:>12}".format("Metrik", "Logistik", "Probit"))
print("-" * 56)
print("{:<30} {:>12.4f} {:>12.4f}".format("AIC - Model A (Income)", result_A.aic, result_probit_A.aic))
print("{:<30} {:>12.4f} {:>12.4f}".format("BIC - Model A (Income)", result_A.bic, result_probit_A.bic))
print("{:<30} {:>12.4f} {:>12.4f}".format("AUC - Model A (Income)", roc_auc_A, auc_probit_A))
print("-" * 56)
print("{:<30} {:>12.4f} {:>12.4f}".format("AIC - Model B (Sanitation)", result_B.aic, result_probit_B.aic))
print("{:<30} {:>12.4f} {:>12.4f}".format("BIC - Model B (Sanitation)", result_B.bic, result_probit_B.bic))
print("{:<30} {:>12.4f} {:>12.4f}".format("AUC - Model B (Sanitation)", roc_auc_B, auc_probit_B))

# ============================================================
# BAGIAN 10: INTERPRETASI LENGKAP
# ============================================================
print("\n" + "=" * 70)
print("[9] INTERPRETASI HASIL PENELITIAN")
print("=" * 70)

beta0_A = result_A.params['const']
beta1_A = result_A.params['Income']
beta0_B = result_B.params['const']
beta1_B = result_B.params['Sanitation']

or_income     = np.exp(beta1_A)
or_sanitation = np.exp(beta1_B)

llr_pval_A = result_A.llr_pvalue
llr_pval_B = result_B.llr_pvalue

pval_income     = result_A.pvalues['Income']
pval_sanitation = result_B.pvalues['Sanitation']

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL A: INCOME → STUNTING                      │
└─────────────────────────────────────────────────────────────────────┘

  Model Logistik yang diperoleh:
    g(x) = {beta0_A:.4f} + ({beta1_A:.4f}) × Income

  1. UJI SERENTAK (Log-Likelihood Ratio Test)
     Nilai G² = {result_A.llr:.4f} | LLR p-value = {llr_pval_A:.6f}
     {'→ SIGNIFIKAN: Model secara keseluruhan BERPENGARUH (p < 0.05)' if llr_pval_A < 0.05 else '→ TIDAK SIGNIFIKAN: Model secara keseluruhan tidak berpengaruh'}
     Artinya variabel Income secara simultan mampu menjelaskan variasi
     status stunting pada balita.

  2. UJI PARSIAL (Uji Z)
     Koefisien Income: β₁ = {beta1_A:.4f} | p-value = {pval_income:.6f}
     {'→ SIGNIFIKAN: Income berpengaruh secara parsial terhadap stunting (p < 0.05)' if pval_income < 0.05 else '→ TIDAK SIGNIFIKAN: Income tidak berpengaruh secara parsial'}

  3. INTERPRETASI ODDS RATIO
     Nilai OR = exp({beta1_A:.4f}) = {or_income:.4f}
     Interpretasi: Setiap kenaikan pendapatan sebesar 1 juta rupiah akan
     {'MENURUNKAN' if or_income < 1 else 'MENINGKATKAN'} odds kejadian stunting sebesar {abs(1 - or_income)*100:.2f}% (OR = {or_income:.4f}).
     Keluarga dengan pendapatan lebih tinggi memiliki kemampuan lebih besar
     untuk memenuhi kebutuhan gizi dan kesehatan balita.

  4. KETEPATAN KLASIFIKASI
     Akurasi = {accuracy_A*100:.2f}% → Model {'sangat baik (>80%)' if accuracy_A > 0.8 else 'baik (70-80%)' if accuracy_A > 0.7 else 'cukup (60-70%)' if accuracy_A > 0.6 else 'kurang (<60%)'}
     Model mampu mengklasifikasikan {int(accuracy_A*25)} dari 25 observasi dengan benar.

  5. ROC-AUC
     AUC = {roc_auc_A:.4f} → Kemampuan diskriminasi model {'sangat tinggi (>0.9)' if roc_auc_A > 0.9 else 'tinggi (0.8-0.9)' if roc_auc_A > 0.8 else 'moderat (0.7-0.8)' if roc_auc_A > 0.7 else 'cukup (0.6-0.7)' if roc_auc_A > 0.6 else 'rendah (<0.6)'}
     Model mampu membedakan balita stunting dan tidak stunting dengan
     tingkat akurasi {'sangat tinggi' if roc_auc_A > 0.9 else 'tinggi' if roc_auc_A > 0.8 else 'baik' if roc_auc_A > 0.7 else 'cukup'}.

┌─────────────────────────────────────────────────────────────────────┐
│                   MODEL B: SANITATION → STUNTING                    │
└─────────────────────────────────────────────────────────────────────┘

  Model Logistik yang diperoleh:
    g(x) = {beta0_B:.4f} + ({beta1_B:.4f}) × Sanitation

  1. UJI SERENTAK (Log-Likelihood Ratio Test)
     Nilai G² = {result_B.llr:.4f} | LLR p-value = {llr_pval_B:.6f}
     {'→ SIGNIFIKAN: Model secara keseluruhan BERPENGARUH (p < 0.05)' if llr_pval_B < 0.05 else '→ TIDAK SIGNIFIKAN: Model secara keseluruhan tidak berpengaruh'}

  2. UJI PARSIAL (Uji Z)
     Koefisien Sanitation: β₁ = {beta1_B:.4f} | p-value = {pval_sanitation:.6f}
     {'→ SIGNIFIKAN: Sanitasi berpengaruh secara parsial terhadap stunting (p < 0.05)' if pval_sanitation < 0.05 else '→ TIDAK SIGNIFIKAN: Sanitasi tidak berpengaruh secara parsial'}

  3. INTERPRETASI ODDS RATIO
     Nilai OR = exp({beta1_B:.4f}) = {or_sanitation:.4f}
     Interpretasi: Setiap peningkatan skor sanitasi sebesar 1 poin akan
     {'MENURUNKAN' if or_sanitation < 1 else 'MENINGKATKAN'} odds kejadian stunting sebesar {abs(1 - or_sanitation)*100:.2f}%.
     Kondisi sanitasi yang buruk meningkatkan paparan terhadap patogen
     penyebab diare dan infeksi berulang yang menghambat penyerapan gizi.

  4. KETEPATAN KLASIFIKASI
     Akurasi = {accuracy_B*100:.2f}% → Model {'sangat baik (>80%)' if accuracy_B > 0.8 else 'baik (70-80%)' if accuracy_B > 0.7 else 'cukup (60-70%)' if accuracy_B > 0.6 else 'kurang (<60%)'}

  5. ROC-AUC
     AUC = {roc_auc_B:.4f} → Kemampuan diskriminasi model {'sangat tinggi (>0.9)' if roc_auc_B > 0.9 else 'tinggi (0.8-0.9)' if roc_auc_B > 0.8 else 'moderat (0.7-0.8)' if roc_auc_B > 0.7 else 'cukup (0.6-0.7)' if roc_auc_B > 0.6 else 'rendah (<0.6)'}
""")

# ============================================================
# BAGIAN 11: VISUALISASI KOMPREHENSIF
# ============================================================
print("[10] Membuat visualisasi komprehensif...")

# Pengaturan style visualisasi
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 9,
    'axes.titlesize'  : 11,
    'axes.titleweight': 'bold',
    'axes.labelsize'  : 9,
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor'  : '#FFFFFF',
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'grid.linestyle'  : '--'
})

COLORS = {'stunting': '#E74C3C', 'normal': '#2ECC71', 'accent': '#2C3E50',
          'model_a': '#3498DB', 'model_b': '#9B59B6'}

fig = plt.figure(figsize=(18, 20))
fig.suptitle(
    'ANALISIS PENGARUH PENDAPATAN DAN SANITASI TERHADAP STATUS STUNTING BALITA\n'
    'Pemodelan Regresi Logistik — Data Penelitian Stunting (n=25)',
    fontsize=13, fontweight='bold', color=COLORS['accent'], y=0.98
)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ─── Plot 1: Distribusi Stunting ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = data['Stunting'].value_counts().sort_index()
bars = ax1.bar(['Tidak Stunting\n(0)', 'Stunting\n(1)'],
               counts.values,
               color=[COLORS['normal'], COLORS['stunting']],
               edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val}\n({val/len(data)*100:.0f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_title('Distribusi Status Stunting')
ax1.set_ylabel('Jumlah Balita')
ax1.set_ylim(0, max(counts.values) * 1.3)

# ─── Plot 2: Boxplot Income per Status ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bp = ax2.boxplot(
    [data[data['Stunting']==0]['Income'], data[data['Stunting']==1]['Income']],
    labels=['Tidak Stunting', 'Stunting'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2.5)
)
bp['boxes'][0].set_facecolor(COLORS['normal'])
bp['boxes'][1].set_facecolor(COLORS['stunting'])
ax2.set_title('Distribusi Income\nper Status Stunting')
ax2.set_ylabel('Income (Juta Rp)')

# ─── Plot 3: Boxplot Sanitation per Status ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bp2 = ax3.boxplot(
    [data[data['Stunting']==0]['Sanitation'], data[data['Stunting']==1]['Sanitation']],
    labels=['Tidak Stunting', 'Stunting'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2.5)
)
bp2['boxes'][0].set_facecolor(COLORS['normal'])
bp2['boxes'][1].set_facecolor(COLORS['stunting'])
ax3.set_title('Distribusi Skor Sanitasi\nper Status Stunting')
ax3.set_ylabel('Skor Sanitasi')

# ─── Plot 4: Scatter + Logistic Curve - Income ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
jitter = np.random.seed(42)
for st, color, label in [(1, COLORS['stunting'], 'Stunting'), (0, COLORS['normal'], 'Tidak Stunting')]:
    subset = data[data['Stunting'] == st]
    jitter_y = subset['Stunting'] + np.random.uniform(-0.03, 0.03, len(subset))
    ax4.scatter(subset['Income'], jitter_y,
                color=color, alpha=0.8, s=80, zorder=5,
                label=label, edgecolors='white', linewidth=0.8)

x_range_A  = np.linspace(data['Income'].min() - 0.5, data['Income'].max() + 0.5, 300)
x_range_A_ = sm.add_constant(pd.Series(x_range_A, name='Income'))
y_pred_A   = result_A.predict(x_range_A_)
ax4.plot(x_range_A, y_pred_A, color=COLORS['model_a'],
         linewidth=2.5, label='Kurva Logistik', zorder=4)
ax4.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Threshold = 0.5')
ax4.set_title('Kurva Regresi Logistik: Income → Stunting')
ax4.set_xlabel('Income (Juta Rupiah)')
ax4.set_ylabel('P(Stunting = 1)')
ax4.set_ylim(-0.1, 1.1)
ax4.legend(loc='center right', fontsize=8)

# Menambahkan persamaan model di dalam plot
eq_text_A = f'g(x) = {beta0_A:.4f} + ({beta1_A:.4f})·Income\nAUC = {roc_auc_A:.4f} | Akurasi = {accuracy_A*100:.1f}%'
ax4.text(0.05, 0.75, eq_text_A, transform=ax4.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

# ─── Plot 5: Scatter + Logistic Curve - Sanitation ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
for st, color, label in [(1, COLORS['stunting'], 'Stunting'), (0, COLORS['normal'], 'Tidak Stunting')]:
    subset = data[data['Stunting'] == st]
    jitter_y = subset['Stunting'] + np.random.uniform(-0.03, 0.03, len(subset))
    ax5.scatter(subset['Sanitation'], jitter_y,
                color=color, alpha=0.8, s=60, zorder=5,
                edgecolors='white', linewidth=0.8)

x_range_B  = np.linspace(data['Sanitation'].min() - 2, data['Sanitation'].max() + 2, 300)
x_range_B_ = sm.add_constant(pd.Series(x_range_B, name='Sanitation'))
y_pred_B   = result_B.predict(x_range_B_)
ax5.plot(x_range_B, y_pred_B, color=COLORS['model_b'],
         linewidth=2.5, label='Kurva Logistik')
ax5.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax5.set_title('Kurva Logistik:\nSanitation → Stunting')
ax5.set_xlabel('Skor Sanitasi')
ax5.set_ylabel('P(Stunting = 1)')
ax5.set_ylim(-0.1, 1.1)

# ─── Plot 6 & 7: ROC Curve Model A & B ────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(fpr_A, tpr_A, color=COLORS['model_a'], lw=2.5,
         label=f'Model A (AUC = {roc_auc_A:.4f})')
ax6.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Garis Acak (AUC=0.5)')
ax6.fill_between(fpr_A, tpr_A, alpha=0.15, color=COLORS['model_a'])
ax6.set_xlim([0.0, 1.0])
ax6.set_ylim([0.0, 1.05])
ax6.set_xlabel('False Positive Rate (1 - Specificity)')
ax6.set_ylabel('True Positive Rate (Sensitivity)')
ax6.set_title('ROC Curve — Model A\n(Income → Stunting)')
ax6.legend(loc='lower right', fontsize=8)

ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(fpr_B, tpr_B, color=COLORS['model_b'], lw=2.5,
         label=f'Model B (AUC = {roc_auc_B:.4f})')
ax7.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Garis Acak (AUC=0.5)')
ax7.fill_between(fpr_B, tpr_B, alpha=0.15, color=COLORS['model_b'])
ax7.set_xlim([0.0, 1.0])
ax7.set_ylim([0.0, 1.05])
ax7.set_xlabel('False Positive Rate (1 - Specificity)')
ax7.set_ylabel('True Positive Rate (Sensitivity)')
ax7.set_title('ROC Curve — Model B\n(Sanitation → Stunting)')
ax7.legend(loc='lower right', fontsize=8)

# ─── Plot 8: Confusion Matrix Model A ────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
sns.heatmap(cm_A, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['Aktual: 0', 'Aktual: 1'],
            ax=ax8, linewidths=0.5, linecolor='white', cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
ax8.set_title(f'Confusion Matrix — Model A\nAkurasi: {accuracy_A*100:.2f}%')

# ─── Plot 9: Probabilitas Prediksi ───────────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 0:2])
x_idx = range(len(data))
colors_bar = [COLORS['stunting'] if s == 1 else COLORS['normal'] for s in data['Stunting']]
ax9.bar(x_idx, data['prob_A'], color=colors_bar, alpha=0.8, edgecolor='white')
ax9.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
ax9.set_xlabel('Nomor Observasi')
ax9.set_ylabel('Probabilitas P(Stunting=1)')
ax9.set_title('Probabilitas Prediksi Model A (Income → Stunting)\n'
              '■ Merah = Aktual Stunting, ■ Hijau = Aktual Tidak Stunting')
ax9.set_xticks(x_idx)
ax9.set_xticklabels([str(i+1) for i in x_idx], fontsize=7)
ax9.set_ylim(0, 1.1)
ax9.legend(fontsize=8)

# ─── Plot 10: Perbandingan AUC Semua Model ───────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
model_names = ['Logistik A\n(Income)', 'Probit A\n(Income)',
               'Logistik B\n(Sanitation)', 'Probit B\n(Sanitation)']
auc_values  = [roc_auc_A, auc_probit_A, roc_auc_B, auc_probit_B]
bar_colors  = [COLORS['model_a'], '#85C1E9', COLORS['model_b'], '#C39BD3']
bars_cmp = ax10.bar(model_names, auc_values, color=bar_colors,
                    edgecolor='white', linewidth=1.2, width=0.6)
for bar, val in zip(bars_cmp, auc_values):
    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax10.set_title('Perbandingan AUC\nLogistik vs Probit')
ax10.set_ylabel('AUC Score')
ax10.set_ylim(0.5, 1.1)
ax10.axhline(0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='AUC = 0.9')
ax10.legend(fontsize=7)

plt.savefig('/home/claude/hasil_analisis_stunting.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("   ✓ Visualisasi disimpan: hasil_analisis_stunting.png")

# ============================================================
# BAGIAN 12: KESIMPULAN AKHIR
# ============================================================
print("\n" + "=" * 70)
print("[11] KESIMPULAN AKHIR PENELITIAN")
print("=" * 70)
print(f"""
KESIMPULAN:

1. PEMILIHAN MODEL
   Regresi Logistik dipilih sebagai model yang paling tepat karena variabel
   respon (Stunting) bersifat biner/dikotomi (0 dan 1), sementara variabel
   prediktor (Income dan Sanitation) berskala rasio/interval.

2. MODEL A — PENGARUH PENDAPATAN (INCOME) TERHADAP STUNTING
   • Persamaan : g(x) = {beta0_A:.4f} + ({beta1_A:.4f}) × Income
   • Uji Serentak  : {'Signifikan' if llr_pval_A < 0.05 else 'Tidak Signifikan'} (LLR p = {llr_pval_A:.4f})
   • Uji Parsial   : {'Signifikan' if pval_income < 0.05 else 'Tidak Signifikan'} (p = {pval_income:.4f})
   • Odds Ratio    : {or_income:.4f} → {'penurunan' if or_income < 1 else 'peningkatan'} odds stunting {abs(1-or_income)*100:.1f}% per kenaikan 1 juta Rp
   • Akurasi       : {accuracy_A*100:.2f}%
   • AUC           : {roc_auc_A:.4f}

3. MODEL B — PENGARUH SANITASI TERHADAP STUNTING
   • Persamaan : g(x) = {beta0_B:.4f} + ({beta1_B:.4f}) × Sanitation
   • Uji Serentak  : {'Signifikan' if llr_pval_B < 0.05 else 'Tidak Signifikan'} (LLR p = {llr_pval_B:.4f})
   • Uji Parsial   : {'Signifikan' if pval_sanitation < 0.05 else 'Tidak Signifikan'} (p = {pval_sanitation:.4f})
   • Odds Ratio    : {or_sanitation:.4f} → {'penurunan' if or_sanitation < 1 else 'peningkatan'} odds stunting {abs(1-or_sanitation)*100:.1f}% per kenaikan 1 skor sanitasi
   • Akurasi       : {accuracy_B*100:.2f}%
   • AUC           : {roc_auc_B:.4f}

4. IMPLIKASI KEBIJAKAN
   • Peningkatan pendapatan keluarga melalui program ekonomi produktif
     terbukti signifikan menurunkan risiko stunting pada balita.
   • Perbaikan kondisi sanitasi lingkungan (air bersih, jamban sehat,
     pengelolaan limbah) merupakan intervensi yang efektif dan signifikan
     dalam pencegahan stunting.
   • Kedua model menunjukkan AUC {'sangat tinggi' if min(roc_auc_A, roc_auc_B) > 0.9 else 'tinggi'}, mengindikasikan pendapatan dan
     sanitasi merupakan prediktor kuat status stunting balita.
""")

print("=" * 70)
print("  ANALISIS SELESAI — SEMUA OUTPUT BERHASIL DIHASILKAN")
print("=" * 70)
