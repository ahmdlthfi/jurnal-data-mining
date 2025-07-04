import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# Membaca Dataset
# ========================
df1 = pd.read_csv('C:/Users/dell/Downloads/tugas_data_mining/sektor_ekonomi.csv')
df2 = pd.read_csv('C:/Users/dell/Downloads/tugas_data_mining/pembiayaan_tahunan.csv')

df1.rename(columns={
    'SEKTOR EKONOMI': 'Sektor',
    'Plafon (Rp juta)': 'Plafon',
    'Outstanding (Rp juta)': 'Outstanding'
}, inplace=True)

# ========================
# Penjelasan Dataset
# ========================
print("=== Penjelasan Mengenai Dataset ===")
print("Plafon       : Jumlah maksimum dana yang dialokasikan ke sektor (Rp juta).")
print("Outstanding  : Dana yang masih berjalan atau belum lunas (Rp juta).")
print("Debitur      : Jumlah penerima dana (perorangan atau badan).")
print("UMKM         : Total pembiayaan ke usaha mikro, kecil, dan menengah.")
print("Usaha_Besar  : Total pembiayaan ke perusahaan besar.\n")
print("Catatan: Meski dua dataset berbeda (berbasis sektor dan tahun), hasil clustering bisa dibandingkan secara tidak langsung melalui tren pembiayaan dan karakteristik sektor.\n")

# ========================
# Fungsi KMeans
# ========================
def run_kmeans(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    return labels, inertia, silhouette, db_index

# ========================
# Clustering Sektor Ekonomi
# ========================
X1 = df1[['Plafon', 'Outstanding', 'Debitur']]
labels1, inertia1, silhouette1, db1 = run_kmeans(X1, 3)
df1['Cluster'] = labels1

# ========================
# Clustering Pembiayaan Tahunan
# ========================
X2 = df2[['UMKM', 'Usaha_Besar']]
labels2, inertia2, silhouette2, db2 = run_kmeans(X2, 2)
df2['Cluster'] = labels2

# ========================
# Evaluasi dan Interpretasi
# ========================
print("=== Evaluasi Sektor Ekonomi ===")
print(f"Jumlah data              : {len(df1)} sektor")
print(f"Jumlah klaster           : 3")
print(f"Inertia                  : {inertia1:.2f}")
print(f"Silhouette Score         : {silhouette1:.4f}")
print(f"Davies-Bouldin Index     : {db1:.4f}")
print("\nInterpretasi:")
for i in range(3):
    sektor = ", ".join(df1[df1['Cluster'] == i]['Sektor'])
    print(f"- Klaster {i}: {sektor}")

print("\n=== Evaluasi Pembiayaan Tahunan ===")
print(f"Jumlah data              : {len(df2)} tahun")
print(f"Jumlah klaster           : 2")
print(f"Inertia                  : {inertia2:.2f}")
print(f"Silhouette Score         : {silhouette2:.4f}")
print(f"Davies-Bouldin Index     : {db2:.4f}")
print("\nInterpretasi:")
for i in range(2):
    tahun = ", ".join(df2[df2['Cluster'] == i]['Tahun'].astype(str))
    print(f"- Klaster {i}: Tahun {tahun}")

# ========================
# Visualisasi Hasil Klaster
# ========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df1, x='Plafon', y='Outstanding', hue='Cluster', palette='Set2')
plt.title("Cluster Sektor Ekonomi")
plt.xlabel("Plafon (Rp juta)")
plt.ylabel("Outstanding (Rp juta)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df2, x='UMKM', y='Usaha_Besar', hue='Cluster', palette='Set1')
plt.title("Cluster Pembiayaan Tahunan")
plt.xlabel("Pembiayaan UMKM")
plt.ylabel("Pembiayaan Usaha Besar")

plt.tight_layout()
plt.show()

# ========================
# Keterkaitan Tidak Langsung Antar Dataset
# ========================
print("\n=== Indikasi Keterkaitan Antar Dataset ===")
top_debitur = df1.sort_values('Debitur', ascending=False).head(5)[['Sektor', 'Debitur']]
print("Sektor dengan debitur terbesar (indikasi sektor UMKM dominan):")
print(top_debitur.to_string(index=False))

avg_UMKM = df2['UMKM'].mean()
avg_usaha_besar = df2['Usaha_Besar'].mean()
proporsi_umkm = avg_UMKM / (avg_UMKM + avg_usaha_besar)

print(f"\nRata-rata pembiayaan UMKM per tahun       : {avg_UMKM:.0f}")
print(f"Rata-rata pembiayaan Usaha Besar per tahun: {avg_usaha_besar:.0f}")
print(f"Proporsi rata-rata pembiayaan untuk UMKM  : {proporsi_umkm*100:.2f}%")

# ========================
# Kesimpulan Otomatis
# ========================
print("\n=== Kesimpulan ===")

# Kualitas klaster sektor
if silhouette1 >= 0.5:
    print(f"- Klaster sektor ekonomi cukup solid (Silhouette={silhouette1:.2f}), menunjukkan pemisahan berdasarkan plafon, outstanding, dan debitur.")
else:
    print(f"- Klaster sektor ekonomi masih lemah (Silhouette={silhouette1:.2f}), bisa dipertimbangkan optimasi klaster lebih lanjut.")

# Kualitas klaster tahunan
if silhouette2 >= 0.5:
    print(f"- Klaster pembiayaan tahunan juga menunjukkan pemisahan baik antara tren UMKM dan usaha besar (Silhouette={silhouette2:.2f}).")
else:
    print(f"- Klaster tahunan belum optimal (Silhouette={silhouette2:.2f}).")

# Proporsi UMKM
if proporsi_umkm > 0.9:
    print(f"- Sekitar {proporsi_umkm*100:.1f}% pembiayaan dialokasikan ke UMKM, menunjukkan dominasi jelas pembiayaan mikro dan kecil.")
else:
    print(f"- Pembiayaan UMKM hanya sekitar {proporsi_umkm*100:.1f}%, belum menunjukkan dominasi.")

# Sektor padat debitur
sektor_padat = ", ".join(top_debitur['Sektor'].tolist()[:3])
print(f"- Sektor dengan debitur tertinggi seperti {sektor_padat} mengindikasikan sektor UMKM dominan.")

# Simpulan akhir
print("- Secara keseluruhan, KMeans berhasil mengungkap pola tersembunyi pada kedua dataset, yang saling mendukung dalam narasi makro pembiayaan UMKM di Indonesia.")