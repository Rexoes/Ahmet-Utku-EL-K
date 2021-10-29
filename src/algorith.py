# Ahmet Utku ELİK
# 5518123001
# Veri Madenciliği Dersi 1. Ödev

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#8. kolon değerlerinden nan olanlar -1 değerine sayısallaştırılıyor 
#Diğer kolonlar için bu işlem yapılırsa olası bir matematiksel hata önlenmiş olur
def nanDegerleriSayisallastir(col):
    count = 0
    for i in range(len(col)):
         if(str(col[i]) == "nan"):
             col[i] = -1
             count += 1
    # print("Kayıp Veri Adedi : {}".format(count))
    return col

def aritmetikOrtBul(col):
    aritmetikOrt = 0
    kayipVeriAdedi = 0
    for i in range(len(col)):
        if(col[i] != -1):
            aritmetikOrt += col[i]
        else :
            kayipVeriAdedi += 1
    aritmetikOrt = aritmetikOrt / (len(col) - kayipVeriAdedi)
    return int(aritmetikOrt)

def standartSapmaHesapla(col):
    aritmetikOrtalama = aritmetikOrtBul(col)
    varyans = 0

    for i in range(len(col)):
        if(col[i] != -1):
            varyans += np.power((col[i] - aritmetikOrtalama), 2)
    varyans /= len(col) - 1
    standartSapma = int(np.sqrt(varyans))
    return standartSapma

def kolonVerileriKaybet(col):
    aritmetikOrtalama = aritmetikOrtBul(col)
    standartSapma = standartSapmaHesapla(col)
    altDeger = aritmetikOrtalama - 2 * standartSapma
    ustDeger = aritmetikOrtalama + 2 * standartSapma
    kaybedilenVeriler = []
    print("Aritmetik Ortalama : {} Standart Sapma : {} Alt Deger : {} Ust Deger : {}".format(aritmetikOrtalama, standartSapma, altDeger, ustDeger))
    for i in range(len(col)):
        if(col[i] != -1): #Okunan değer kayıp veri değil
            if(col[i] < altDeger or col[i] > ustDeger): #Ve aralık dışında ise
                kaybedilenVeriler.append(col[i])
                col[i] = -1
    # print("Kaybedilen Veri Adedi : {}".format(len(kaybedilenVeriler)))
    # for i in range(len(kaybedilenVeriler) - 5):
    #     print("[{}]  [{}]  [{}]  [{}]  [{}]".format(col[i], col[i+1], col[i+2], col[i+3], col[i+4]))
    #     i+=4
    return col

def kayipVerileriDoldur(col):
    aritmetikOrtalama = aritmetikOrtBul(col)
    for i in range(len(col)):
        if(col[i] == -1):
            col[i] = aritmetikOrtalama
    return col

def kayipVeriVarmi(col):
    for i in range(len(col)):
        if(col[i] == -1):
            return True
    return False

def hangiIndexte(liste, veri):
    for i in range(len(liste)):
        if(liste[i] == veri):
            return i
    return -1 #Veri listede yok

def kategoriSayisallastir(col, kategoriIsimleri):
    for i in range(len(col)):
        col[i] = hangiIndexte(kategoriIsimleri, str(col[i]))
    return col

def kolonKategoriAdetHesapla(col, countArray): # sayısallaştırılmış kolon
    for i in range(len(col)):                     # gelme şartıyla
        countArray[col[i]] += 1;
    return countArray

def dizideMaxBul(array):
    temp = array[0]
    for i in range(len(array)):
        if(array[i] > temp):
            temp = array[i]
    return temp

def dizideMinBul(array):
    temp = array[0]
    for i in range(len(array)):
        if(array[i] < temp):
            temp = array[i]
    return temp

def azRastlananaÇokRastlananıAta(col, kategoriAdetleri):
    azRastlanan = dizideMinBul(kategoriAdetleri)                                   #Az rastlanan sınıfın kategorideki adedi
    azRastlananIndex = hangiIndexte(kategoriAdetleri, azRastlanan)                #Az rastlanan sınıf
    
    cokRastlanan = dizideMaxBul(kategoriAdetleri)                                  #Çok rastlanan sınıfın kategorideki adedi
    cokRastlananIndex = hangiIndexte(kategoriAdetleri, cokRastlanan)               #Çok rastlanan sınıf
    for i in range(len(col)):
        if(col[i] == azRastlananIndex):
            col[i] = cokRastlananIndex
    return col

def entropiHesapla(strokeArray):
    toplamAdet = strokeArray[0] + strokeArray[1]
    entropi = 0
    if(strokeArray[0] != 0):
        entropi += strokeArray[0] / toplamAdet * np.log(toplamAdet / strokeArray[0])
    if(strokeArray[1] != 0):
        entropi += strokeArray[1] / toplamAdet * np.log(toplamAdet / strokeArray[1])
    return entropi #float tip döner

def sinifEntropiHesapla(classValue):
    val0 = 0
    val1 = 0
    for i in classValue:
        if(i == 1):
            val1 += 1
    rowCount = len(classValue)
    val0 = rowCount - val1
    sinifEntropi = val0 / rowCount * np.log(rowCount / val0) + val1 / rowCount * np.log(rowCount / val1)
    print("1 lerin Adedi : {}  0 ların adedi : {}  Toplam : {}".format(val1, val0, rowCount))
    return sinifEntropi
    
def kolonInformationGainHesapla(calculationCol, classValue, kategoriAdedi): #Sayısallaştırılmış kolon gönderilmeli!
    entropiDegerleri = [0,0,0,0,0]
    stroke = [0,0] #Sınıf bilgisi için 0. indexte kalp krizi geçirmeyen, 1. indexte ise kalp krizi geçiren adedini saklıyacak.
    adet = [0,0,0,0,0]#Örneğin Gender kolonu için erkek ve kadın adedini tutmak için esnek boyutlu değişken
    count = 0
    for index in range(kategoriAdedi):
        for i in range(len(calculationCol)):
            if(calculationCol[i] == index):
                count += 1
                if(classValue[i] == 1): #Kalp krizi geçirmiş ise
                    stroke[1] += 1      #Karşılık gelen stroke[1] değişkenini artır
                else :                  #Kalp krizi geçirmemiş ise
                    stroke[0] += 1
        adet[index] = count
        entropiDegerleri[index] = entropiHesapla(stroke)
        stroke[0] = 0
        stroke[1] = 0
        count = 0
    totalEntropi = 0
    for indis in range(kategoriAdedi):
        totalEntropi += adet[indis] * entropiDegerleri[indis] / len(calculationCol)
        
    sinifEntropi = sinifEntropiHesapla(classValue)
    print("Sınıf Entropi Değeri : {}".format(sinifEntropi))
    return (sinifEntropi - totalEntropi)

def icindeVarmi(array, veri):
    for i in array:
        if(i == veri):
            return True
    return False

def reverseArray(array):
    tempArray = []
    arrayLen = len(array)
    for i in range(arrayLen):
        tempArray.append(array[arrayLen - i - 1])
    array = tempArray
    return array

def informationGain2Entropi(array, systemEntropy):                              #Entropy = System Entropy - InformationGain
    tempArray = []
    for i in range(len(array)):
        tempArray.append((systemEntropy - array[i]))
    return tempArray

#
def oklid_uzaklik(v1,v2):
    col_sayi=len(v1) #v1 vektörünün uzunluğu
    t=0
    for i in range(col_sayi):
        t+=(v1[i]-v2[i])*(v1[i]-v2[i])
    
    return np.sqrt(t) #toplamin karaköküne dönüyor

def ozellik_normallestir(col):
    the_max=np.max(col)
    the_min=np.min(col)
    for i in range(len(col)):
        col[i]=(col[i]-the_min)/(the_max-the_min)#istenirse buraya the_max-the_min'in 0 olma exception'i eklenebilir
    return col
#

bütünVeriler=pd.read_excel("dataset-stroke-data.xlsx")                           #5518123001 % 5 = 1
bütünVeriler_np=np.array(bütünVeriler)                                          #Data Frame'i array formuna dönüştürüyoruz
calisilacakVeriler = []
index = 2                                                                       #5518123001 % 3 = 2
while(True):
    calisilacakVeriler.append(bütünVeriler_np[index])
    index += 3
    if(index > 5110):
        break
calisilacakVeriler = np.array(calisilacakVeriler)
satir_sayisi=calisilacakVeriler.shape[0]
sutun_sayisi=calisilacakVeriler.shape[1]
calisilacakVeriler = calisilacakVeriler[np.random.permutation(satir_sayisi), :] #Çalışılacak veri seti shuffle ediliyor

#Bu kısımda sayısal kolonlardaki nan değerler sayısallaştırıldı ardından
#Ödevin 1. maddesine ugun olarak standart sapma ve merkez değerlerine bağlı olarak
#Sayısal kolonlardan 2 adet veri silinmiştir


#En az görülen değer yerine en fazla görülen değeri yaz
#Kategorik tiplerden en az 3 farklı değer içeren kolonları ele alacağız 5-> workType ve 9-> smokingStatus

#work_type kolonunda en sık rastlanan ile en az rastlanan arasında bariz fark olduğu için bu kolonda veri değiştirmeye gidildi
#test(temp[:,5])


sayısalSütünlar = (1,7,8)                                                       #Sayısal kolon indexleri
temp = calisilacakVeriler
for i in sayısalSütünlar:
    temp[:,i] = nanDegerleriSayisallastir(temp[:,i])
    temp[:,i] = kolonVerileriKaybet(temp[:,i])

                                                                                    #Kayıp verileri doldurma, kayıp veriler 1,7 ve 8. kolonda yer almaktadır.
for i in sayısalSütünlar:
    temp[:,i] = kayipVerileriDoldur(temp[:,i])

work_type = ("children", "Govt_job", "Never_worked", "Private", "Self-employed")
temp[:, 5] = kategoriSayisallastir(temp[:, 5], work_type)
work_type_count = [0,0,0,0,0]
work_type_count = kolonKategoriAdetHesapla(temp[:,5], work_type_count)

                                                                                    #work_tyoe kolonundaki veri kategorileri
smoking_status = ("formerly smoked", "never smoked", "smokes", "Unknown")            #smoking_status kolonundaki veri kategorileri
temp[:, 9] = kategoriSayisallastir(temp[:, 9], smoking_status)                      #smoking_status kolonu sayısallaştırma
smoking_status_count = [0,0,0,0]
                  #Kategorilerin adetleri work_type ve smoking_status listelerinin index karşılıklarına atanıyor
smoking_status_count = kolonKategoriAdetHesapla(temp[:,9], smoking_status_count)

x0 = temp[:,0]
x2 = temp[:,2]
x3 = temp[:,3]
x4 = temp[:,4]
x6 = temp[:,6]
x5 = temp[:,5]
x9 = temp[:,9]

plt.style.use('ggplot')
plt.hist(x0, bins=20)
plt.show()
plt.hist(x2, bins=20)
plt.show()
plt.hist(x3, bins=20)
plt.show()
plt.hist(x4, bins=20)
plt.show()
plt.hist(x6, bins=20)
plt.show()
plt.hist(x5, bins=20)
plt.show()
plt.hist(x9, bins=20)
plt.show()

temp[:, 5] = azRastlananaÇokRastlananıAta(temp[:, 5], work_type_count)                #work_type kolonundaki az rastlanan kategoriyi çok rastlanan ile değiştirme
x5 = temp[:,5]
plt.hist(x5, bins=20)
plt.show()     
work_type_count = [0,0,0,0,0]
work_type_count = kolonKategoriAdetHesapla(temp[:,5], work_type_count)                                                                           #Never_worked = Private yapmakta 

#3. Madde Information Gain
                                      #Kategorik yapıda 2 veri durumu olan kolonlar
gender = ("Female", "Male")
ever_married = ("No", "Yes")
Residence_type = ("Urban", "Rural")
temp[:, 0] = kategoriSayisallastir(temp[:, 0], gender)#Kategorik tipteki kolonlar
temp[:, 4] = kategoriSayisallastir(temp[:, 4], ever_married)#Information Gain işlemi
temp[:, 6] = kategoriSayisallastir(temp[:, 6], Residence_type)#Yapılabilmesi için sayısallaştırılıyor

kategorikTiptekiKolonlar = (0, 2, 3, 4, 5, 6, 9)  
kolonInformationGainDegerleri = [0,0,0,0,0,0,0,0,0,0]                                                #Kategorik tipteki kolonlar için
for i in kategorikTiptekiKolonlar:
    if(i == 5):
        kolonInformationGainDegerleri[i] = kolonInformationGainHesapla(temp[:,i], temp[:, 10], 5)    #Kategori adetlerini fonksiyona doğru bir şekilde gönderebilmek için
    elif(i == 9):
        kolonInformationGainDegerleri[i] = kolonInformationGainHesapla(temp[:,i], temp[:, 10], 4)
    else :
        kolonInformationGainDegerleri[i] = kolonInformationGainHesapla(temp[:,i], temp[:, 10], 2)

print("*********************************************************************************************")
for i in kategorikTiptekiKolonlar:
    print("{}. Index'te ki kolonun Information Gain Degeri : {}".format(i, kolonInformationGainDegerleri[i]))
print("*********************************************************************************************")

sortedIndexAboutInformationGainValues = [0,0,0,0,0,0,0]                                            #Kolonların Information Gain Değerleri Büyükten Küçüğe Saklayan Değişken
sortedAboutInformationGainValues = [0,0,0,0,0,0,0]

maxValue = 0
maxValueIndex = 0
index = 0

for i in range(len(kolonInformationGainDegerleri)):
    maxValueIndex = i
    maxValue = kolonInformationGainDegerleri[maxValueIndex]
    j = 0
    while(j < len(kolonInformationGainDegerleri)):
        if(kolonInformationGainDegerleri[j] >= maxValue):
            maxValue = kolonInformationGainDegerleri[j]
            maxValueIndex = j
        j+=1
    if(kolonInformationGainDegerleri[maxValueIndex] > 0):
        sortedIndexAboutInformationGainValues[index] = maxValueIndex
        sortedAboutInformationGainValues[index] = kolonInformationGainDegerleri[maxValueIndex]
        kolonInformationGainDegerleri[maxValueIndex] = -1
        index+=1

for i in range(len(sortedIndexAboutInformationGainValues)):
    print("En Yüksek {}. Information Gain Değeri {}. Indexte Yer Alan Özellikte Degeri : {}".format((i+1), sortedIndexAboutInformationGainValues[i], sortedAboutInformationGainValues[i]))
print("*********************************************************************************************")

#4. Madde Kategorik tipteki özelliklerin entropi değerlerine göre sıralanması

sortedIndexAboutEntropiValues = reverseArray(sortedIndexAboutInformationGainValues)
systemEntropy = sinifEntropiHesapla(temp[:,10])
sortedAboutEntropiValues = informationGain2Entropi(sortedAboutInformationGainValues,systemEntropy)
sortedAboutEntropiValues = reverseArray(sortedAboutEntropiValues)

for i in range(len(sortedIndexAboutEntropiValues)):
    print("En Yüksek {}. Entropi Değeri {}. Indexte Yer Alan Özellikte Degeri : {}".format((i+1), sortedIndexAboutEntropiValues[i], sortedAboutEntropiValues[i]))
print("*********************************************************************************************")

#5 Sayısal tipteki özelliklerden bir veri seti oluşturulacak (1, 7 ve 8. indexteki özellikler)
# Bu veri setine PCA ugyulanarak veri boyutu 2'ye düşürülecek
# Sonra Scatter Plot'ta bu 2 boyutlu veriyi sınıf bilgisine göre renklendirerek çizdir

#Yöntem 1

yeniVeriSeti = bütünVeriler #pd.read_excel("dataset-stroke-data.xlsx")
features = ['age', 'avg_glucose_level', 'bmi'] #Sayısal kolonlar

x = yeniVeriSeti.loc[:, features].values # Separating out the features
y = yeniVeriSeti.loc[:,['stroke']].values # Separating out the target

tempX = []
tempY = []
index = 2
while(True):
    tempX.append(x[index])
    tempY.append(y[index])
    index += 3
    if(index > 5110):
        break
x = np.array(tempX)
sınıfBilgisi = np.array(tempY)
for i in range(3):
    x[:,i] = nanDegerleriSayisallastir(x[:,i])
    x[:,i] = kolonVerileriKaybet(x[:,i])
    x[:,i] = kayipVerileriDoldur(x[:,i])
    
x = StandardScaler().fit_transform(x) # Standardizing the features


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
              , columns = ['principal component 1', 'principal component 2'])

sınıfBilgisiDf = pd.DataFrame(data = sınıfBilgisi, columns = ['stroke'])

finalDf = pd.concat([principalDf, sınıfBilgisiDf[['stroke']]], axis = 1)

#Sırada çizim işlemleri var

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
strokes = [0,1]
colors = ['r', 'g']

for target, color in zip(strokes,colors):
    indicesToKeep = finalDf['stroke'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
ax.legend(strokes)
ax.grid()


#Knn En Yakın Komşu Algoritması
#Başlangıç

stroke = np.array(finalDf)
satir_sayisi=stroke.shape[0]
sutun_sayisi=stroke.shape[1]
stroke_shuffle = stroke[np.random.permutation(satir_sayisi),:]

for i in range(3):
    stroke_shuffle[:,i]=ozellik_normallestir(stroke_shuffle[:,i])
    

#######veri setinin parcalanmasi####

egitim_set=stroke_shuffle[:850,:]
egitim_X=egitim_set[:,:sutun_sayisi-1]
egitim_Y=egitim_set[:,sutun_sayisi-1]
egitim_num=egitim_X.shape[0]

val_set=stroke_shuffle[850:1275,:] 
val_X=val_set[:,:sutun_sayisi-1]
val_Y=val_set[:,sutun_sayisi-1]
val_num=val_X.shape[0]

test_set=stroke_shuffle[1275:,:] 
test_X=test_set[:,:sutun_sayisi-1]
test_Y=test_set[:,sutun_sayisi-1]
test_num=test_X.shape[0]

###############################
#en ideal k degerine karar verebilmek icin validasyon seti uzerinde farkli k degerleri icin 
#en yakin komsu algoritmasini calistiyoruz, hangi k icin en yuksek performansi alacaksak o k degerini 
#test setinde kullanmak uzere sabitliyoruz


aday_k=[1,3,5,7,9,11]
performanslar=[] #her bir k degerinden elde edilcek performans degeri bu listede tutalacak
for k in aday_k: #aday_k listesinin icini geziyor
    tahminler=[]# her bir validasyon ornegiicin urettigimiz sinif tahminini bu listede tutacagiz.
    
    for v in range(val_num):
        sinifi_merak_edilen=val_X[v,:] #bunu siniflandiracagiz
        uzakliklar=[]#bu liste sinifini merak ettigimiz validasyon orneginin tüm egitim örneklerine olan uzakliklarini tutacak
        for e in range(egitim_num): #her bir egitim ornegi icin
            test_edilen=egitim_X[e,:]
            uzaklik=oklid_uzaklik(sinifi_merak_edilen,test_edilen)
            uzakliklar.append(uzaklik) # e. siradaki egitim ornegi ile v. siradaki validasyon ornegi arasiu uzaklik
        en_yakin_komsular=np.argsort(uzakliklar)#egitim örneklerinin sinif merak edilenvalidasyon ornegine yakinliklarina göre siralanmasi
        en_yakin_komsular_siniflar=egitim_Y[en_yakin_komsular[:k]] #egitim örneklerinin ilk k tanesini aliyoruz
        en_cok_gorulen_sinif=stats.mode(en_yakin_komsular_siniflar)[0][0]#en yakin k egitim orneginde en cok gorulen sinif
        tahminler.append(en_cok_gorulen_sinif)
    #bu noktada tum validasyon orneklerini siniflandirmis oluyoruz, simdi bu tahminleri
    #validasyon örneklerinin gercek siniflari ile karsilastiriyoruz
    basari=0   
    for v in range(val_num):
        if tahminler[v]==val_Y[v]:#dogru tahmin ettigimiz her validasyon ornegi icin basari sayimizi bir artiriyoruz.
            basari+=1
    performans=(basari/val_num)*100 #dikkat edersek burda en disardaki for loop'unun icindeyiz, elde edilen bu performans  belirli bir k degeri icin elde edilen performanstir
    performanslar.append(performans)
best_k=aday_k[np.argmax(performanslar)] #np.argmax(performanslar kacinci k degerinde en yuksek performans alindigini verir

####################################
#bundan sonra validasyon setini kullanarak öğrendigimiz k degeri icin test örneklerini yine ayni validasyon setinde oldugu gibi siniflandiriyoruz.

tahminler=[]

for t in range(test_num):
    sinifi_merak_edilen=test_X[t,:]
    uzakliklar=[]
    for e in range(egitim_num):
        test_edilen=egitim_X[e,:]
        uzaklik=oklid_uzaklik(sinifi_merak_edilen,test_edilen)
        uzakliklar.append(uzaklik)
    en_yakin_komsular=np.argsort(uzakliklar)
    en_yakin_komsular_siniflar=egitim_Y[en_yakin_komsular[:best_k]]
    en_cok_gorulen_sinif=stats.mode(en_yakin_komsular_siniflar)[0][0]
    tahminler.append(en_cok_gorulen_sinif)

basari=0   
for t in range(test_num):
    if tahminler[t]==test_Y[t]:
        basari+=1

performans=(basari/test_num)*100     

print("\n")
print("k-En yakin Komsu siniflandirma performansi: {}".format(performans))
