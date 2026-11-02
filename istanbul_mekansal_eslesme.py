# -*- coding: utf-8 -*-
"""İstanbul toplu taşıma–nüfus mekânsal eşleşmesi.
Colab: !pip install -q pandas openpyxl geopandas shapely pyproj folium matplotlib scipy
Dosyalar: istanbul_toplu_tasima_birlesik.xlsx (zorunlu), istanbul_ilceler.geojson (ilçe sınırları; hizmet açığı için zorunlu).
GeoJSON her ilçe için ad içeren bir sütuna sahip olmalı ve CRS bilgisi bulunmalı (yoksa WGS84 varsayılır).
"""
from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point
from scipy.spatial import cKDTree

BASE=Path('/content') if Path('/content/istanbul_toplu_tasima_birlesik.xlsx').exists() else Path('.')
XLS=BASE/'istanbul_toplu_tasima_birlesik.xlsx'
GEO=BASE/'istanbul_ilceler.geojson'
OUT=BASE/'analiz_sonuclari'; OUT.mkdir(exist_ok=True)
# İhtiyaca göre değiştirilebilir eşikler
POP_HIGH_QUANTILE=.75
SUPPLY_LOW_QUANTILE=.25
WALKING_RADIUS_M=500

def norm(s):
    s=str(s).strip().upper().replace('İ','I').replace('ı','I')
    s=''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
    return re.sub(r'[^A-Z0-9]','',s)

def coord(x, kind):
    """Excel aktarımındaki 410.191.700.005.564 gibi nokta-kümelenmiş koordinatları düzelt."""
    if pd.isna(x):return np.nan
    s=str(x).strip().replace(',','.')
    try: v=float(s)
    except ValueError:
        digits=re.sub('[^0-9]','',s)
        if not digits:return np.nan
        v=float(digits)
    lo,hi=(40,42.5) if kind=='lat' else (27,31)
    if lo<=v<=hi:return v
    for k in range(1,17):
        q=v/(10**k)
        if lo<=q<=hi:return q
    return np.nan

print('Excel okunuyor:',XLS)
nuf=pd.read_excel(XLS,sheet_name='Nufus')
stops=pd.read_excel(XLS,sheet_name='Duraklar',usecols=['stop_id','stop_lat','stop_lon'])
routes=pd.read_excel(XLS,sheet_name='Hatlar',usecols=['route_id','route_short_name','route_type'])
trips=pd.read_excel(XLS,sheet_name='Seferler',usecols=['trip_id','route_id','service_id'])
rail=pd.read_excel(XLS,sheet_name='Rayli_Yolcu',usecols=['transaction_year','station_number','station_name','town','longitude','latitude','passanger_cnt'])

# 1) Nüfus: yalnız yaş-cinsiyet hücrelerini topla, toplam sütunları tekrar toplama.
nuf['Yıl']=pd.to_numeric(nuf['Yıl'],errors='coerce')
year=int(nuf['Yıl'].max())
ages=[c for c in nuf if str(c).startswith(('Erkek ve ','Kadın ve '))]
if not ages:raise ValueError('Nufus sayfasında yaş-cinsiyet sütunları bulunamadı.')
nuf['nufus']=nuf[ages].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=1)
pop=nuf.loc[nuf['Yıl']==year,['İlçe','nufus']].copy()
pop['ilce_key']=pop['İlçe'].map(norm)
pop=pop.groupby('ilce_key',as_index=False).agg(ilce=('İlçe','first'),nufus=('nufus','sum'))
pop.to_csv(OUT/'ilce_nufus.csv',index=False,encoding='utf-8-sig')
print('Nüfus yılı:',year,'İlçe sayısı:',len(pop))

# 2) Durak noktaları: GTFS'de coğrafi durak arzı; stop_times olmadığı için seferleri duraklara bağlamıyoruz.
stops['lat']=stops['stop_lat'].map(lambda x:coord(x,'lat'))
stops['lon']=stops['stop_lon'].map(lambda x:coord(x,'lon'))
stops=stops.dropna(subset=['lat','lon']).drop_duplicates('stop_id')
stops=stops[stops.lat.between(40,42.5)&stops.lon.between(27,31)]
sg=gpd.GeoDataFrame(stops[['stop_id']],geometry=gpd.points_from_xy(stops.lon,stops.lat),crs='EPSG:4326')
print('Geçerli durak:',len(sg))

# 3) Hat-sefer kayıtları: service_id farklı takvimleri temsil eder; gerçek günlük sefer sayısı DEĞİLDİR.
tripcounts=trips.groupby(['route_id','service_id'],dropna=False).size().rename('trip_kaydi').reset_index()
tripcounts=tripcounts.merge(routes,on='route_id',how='left')
tripcounts.to_csv(OUT/'hat_servis_sefer_kayitlari.csv',index=False,encoding='utf-8-sig')

# 4) Raylı istasyonlar ve 2025 yolcu kayıtları: kullanım, arzın kendisi değildir.
rail['lat']=rail['latitude'].map(lambda x:coord(x,'lat'))
rail['lon']=rail['longitude'].map(lambda x:coord(x,'lon'))
rail['passanger_cnt']=pd.to_numeric(rail['passanger_cnt'],errors='coerce')
rail=rail.dropna(subset=['lat','lon'])
rail=rail[rail.lat.between(40,42.5)&rail.lon.between(27,31)]
rail2025=rail[rail.transaction_year.astype(str).str.startswith('2025')].copy()
rs=rail2025.groupby(['station_number','station_name','town'],dropna=False).agg(lat=('lat','median'),lon=('lon','median'),yolcu_kaydi_toplami=('passanger_cnt','sum')).reset_index()
rs.to_csv(OUT/'rayli_istasyonlar.csv',index=False,encoding='utf-8-sig')
rg=gpd.GeoDataFrame(rs,geometry=gpd.points_from_xy(rs.lon,rs.lat),crs='EPSG:4326')

# 5) Harita (sınır olmadan da oluşturulur)
m=folium.Map(location=[41.04,29.00],zoom_start=10,tiles='CartoDB positron')
from folium.plugins import FastMarkerCluster
FastMarkerCluster(sg[['geometry']].assign(lat=sg.geometry.y,lon=sg.geometry.x)[['lat','lon']].values.tolist(),name='Otobüs durakları').add_to(m)
folium.FeatureGroup(name='Raylı istasyonlar',show=True).add_to(m)
rail_layer=folium.FeatureGroup(name='Raylı istasyonlar (noktalar)',show=False)
for _,r in rs.iterrows():
    folium.CircleMarker([r.lat,r.lon],radius=3,color='darkred',fill=True,fill_opacity=.7,
                        tooltip=str(r.station_name)).add_to(rail_layer)
rail_layer.add_to(m)

if not GEO.exists():
    folium.LayerControl().add_to(m)
    m.save(OUT/'ulasim_noktalari.html')
    print('\nUYARI: istanbul_ilceler.geojson bulunamadı. İlçe alanları ve sınırları olmadan nüfus yoğunluğu / hizmet açığı HESAPLANAMAZ.')
    print('Bir İstanbul ilçe sınır GeoJSON dosyasını aynı klasöre istanbul_ilceler.geojson adıyla koyup tekrar çalıştırın.')
else:
    borders=gpd.read_file(GEO)
    if borders.crs is None:
        print('UYARI: GeoJSON CRS tanımsız; EPSG:4326 varsayıldı.')
        borders=borders.set_crs('EPSG:4326')
    candidates=[c for c in borders.columns if norm(c) in {'ILCE','ILCEADI','ILCEISMI','NAME','DISTRICT','AD','ADI'}]
    if not candidates:raise ValueError('GeoJSON ilçe adı sütunu bulunamadı. Mevcut sütunlar: '+str(list(borders.columns)))
    borders['ilce_key']=borders[candidates[0]].map(norm)
    borders=borders[['ilce_key','geometry']].dropna(subset=['geometry']).dissolve(by='ilce_key',as_index=False)
    borders=borders.to_crs('EPSG:32635') # İstanbul metre bazlı UTM 35N
    borders['alan_km2']=borders.area/1e6
    borders=borders.merge(pop,on='ilce_key',how='left',validate='one_to_one')
    missing=borders.loc[borders.nufus.isna(),'ilce_key'].tolist()
    if missing:print('EŞLEŞME UYARISI: nüfus bulunamayan ilçeler:',missing)
    # İdari sınır içinde kalan durak ve istasyon sayıları
    st=gpd.sjoin(sg.to_crs(borders.crs),borders[['ilce_key','geometry']],how='inner',predicate='within')
    rt=gpd.sjoin(rg.to_crs(borders.crs),borders[['ilce_key','geometry']],how='inner',predicate='within')
    sct=st.groupby('ilce_key')['stop_id'].nunique().rename('durak_sayisi')
    rct=rt.groupby('ilce_key')['station_number'].nunique().rename('rayli_istasyon_sayisi')
    borders=borders.join(sct,on='ilce_key').join(rct,on='ilce_key')
    borders[['durak_sayisi','rayli_istasyon_sayisi']]=borders[['durak_sayisi','rayli_istasyon_sayisi']].fillna(0)
    borders['nufus_yogunlugu_km2']=borders.nufus/borders.alan_km2
    borders['onbin_kisi_basina_durak']=borders.durak_sayisi/borders.nufus*10000
    borders['km2_basina_durak']=borders.durak_sayisi/borders.alan_km2
    # İlçe içinde duraktan 500 m uzakta kalan *alan oranı*: nüfus erişim oranı değildir.
    stop_union=st.geometry.buffer(WALKING_RADIUS_M).union_all() if len(st) else None
    # Her ilçe için TÜM duraklardan erişim tamponu (sınır ötesi yakın duraklar dahil)
    all_buffer=sg.to_crs(borders.crs).geometry.buffer(WALKING_RADIUS_M).union_all()
    borders['duraga_500m_uzak_alan_orani']=borders.geometry.apply(lambda g: 1-g.intersection(all_buffer).area/g.area if g.area else np.nan)
    # Hizmet açığı: nüfus yoğunluğu üst çeyrek ve 10 bin kişiye durak alt çeyrek.
    # İlçe ölçeğinde tarama göstergesidir; mahalle düzeyi ya da kapasite ölçümü değildir.
    valid=borders.dropna(subset=['nufus_yogunlugu_km2','onbin_kisi_basina_durak'])
    high=valid.nufus_yogunlugu_km2.quantile(POP_HIGH_QUANTILE)
    low=valid.onbin_kisi_basina_durak.quantile(SUPPLY_LOW_QUANTILE)
    borders['hizmet_acigi']=((borders.nufus_yogunlugu_km2>=high)&(borders.onbin_kisi_basina_durak<=low))
    borders['hizmet_acigi']=borders['hizmet_acigi'].fillna(False)
    cols=['ilce','nufus','alan_km2','nufus_yogunlugu_km2','durak_sayisi','rayli_istasyon_sayisi','onbin_kisi_basina_durak','km2_basina_durak','duraga_500m_uzak_alan_orani','hizmet_acigi']
    summary=borders[cols].sort_values(['hizmet_acigi','nufus_yogunlugu_km2'],ascending=[False,False])
    summary.to_csv(OUT/'ilce_eslesme_ve_hizmet_acigi.csv',index=False,encoding='utf-8-sig')
    summary[summary.hizmet_acigi].to_csv(OUT/'hizmet_acigi_ilceleri.csv',index=False,encoding='utf-8-sig')
    print('\nHİZMET AÇIĞI (ilçe düzeyi adayları):')
    print(summary.loc[summary.hizmet_acigi,['ilce','nufus_yogunlugu_km2','onbin_kisi_basina_durak']].to_string(index=False))
    print(f'Eşikler: yoğunluk >= {high:.1f} kişi/km²; durak <= {low:.2f} / 10 bin kişi')
    plot=borders.to_crs('EPSG:4326')
    folium.GeoJson(plot[['ilce','nufus_yogunlugu_km2','onbin_kisi_basina_durak','hizmet_acigi','geometry']],
       name='Hizmet açığı ilçeleri',
       style_function=lambda feat:{'fillColor':'#dc2626' if feat['properties']['hizmet_acigi'] else '#16a34a',
                                   'color':'#555','weight':1,'fillOpacity':.35},
       tooltip=folium.GeoJsonTooltip(fields=['ilce','nufus_yogunlugu_km2','onbin_kisi_basina_durak','hizmet_acigi'],
          aliases=['İlçe','Kişi/km²','10 bin kişiye durak','Hizmet açığı'])).add_to(m)
    fig,ax=plt.subplots(figsize=(10,8))
    borders.plot(column='nufus_yogunlugu_km2',legend=True,cmap='YlOrRd',ax=ax,edgecolor='gray',linewidth=.3)
    ax.set_title(f'İstanbul ilçe nüfus yoğunluğu ({year})');ax.axis('off');fig.tight_layout()
    fig.savefig(OUT/'nufus_yogunlugu.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.scatter(valid.nufus_yogunlugu_km2,valid.onbin_kisi_basina_durak,alpha=.7)
    ax.axvline(high,color='red',ls='--');ax.axhline(low,color='red',ls='--')
    for _,r in valid[valid.hizmet_acigi].iterrows():ax.annotate(r.ilce,(r.nufus_yogunlugu_km2,r.onbin_kisi_basina_durak),fontsize=8)
    ax.set(xlabel='Nüfus yoğunluğu (kişi/km²)',ylabel='10 bin kişiye düşen durak',title='Nüfus–durak arzı eşleşmesi')
    fig.tight_layout();fig.savefig(OUT/'eslesme_dagilim.png',dpi=180);plt.close(fig)
    # Excel sonuç dosyası
    with pd.ExcelWriter(OUT/'analiz_ozeti.xlsx',engine='openpyxl') as writer:
        summary.to_excel(writer,sheet_name='Ilce_Eslesme',index=False)
        summary[summary.hizmet_acigi].to_excel(writer,sheet_name='Hizmet_Acigi',index=False)
        pop.to_excel(writer,sheet_name='Nufus',index=False)
        tripcounts.to_excel(writer,sheet_name='Sefer_Kayitlari',index=False)
folium.LayerControl().add_to(m)
m.save(OUT/'istanbul_toplu_tasima_haritasi.html')
print('\nBitti. Çıktılar:',OUT.resolve())
