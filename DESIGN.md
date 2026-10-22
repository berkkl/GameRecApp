---
version: 1
name: GameRecApp
lang: en
description: Steam oyun keşfi için nötr yüzeyler, petrol mavisi vurgu ve okunabilir oyun kartları.
colors:
  background: "#fafafa"
  surface: "#ffffff"
  surface-muted: "#f2f4f5"
  text: "#24272b"
  text-muted: "#616970"
  primary: "#185c68"
  primary-hover: "#124954"
  primary-soft: "#eaf3f5"
  border: "#e3e6e8"
  border-control: "#879399"
  success: "#266344"
  success-soft: "#eef6f0"
  warning: "#86531c"
  warning-soft: "#fbf3e6"
  danger: "#a13232"
  danger-soft: "#fef1f1"
typography:
  body: "16px / 1.65"
  interface: "14px / 1.5"
  metadata: "13px"
  page-title: "clamp(28px, 3vw, 40px)"
  section-title: "24px"
  card-title: "20px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "20px"
  6: "24px"
  8: "32px"
  10: "40px"
rounded:
  control: "8px"
  panel: "12px"
---

# GameRecApp tasarım sistemi

Kullanıcı bir oyun isteği yazar, sevdiği oyunları ekler ve Steam oyunlarını inceler. Arama ve sonuçlar aynı sayfada kalır.

## Stil sahipleri

- `public/design.css`: renkler, 4px boşluk ölçeği, yazı ailesi, köşeler, ortak girdi ve klavye odağı kuralları. Keşif, geçmiş ve benchmark sayfaları aynı dosyayı kullanır.
- `public/discover.css`: keşif kabuğu, arama formu, başlangıç seçenekleri ve sonuç kartları.
- `public/sequel.css`: ilişkili oyunların açılır listesi.
- `public/index.html`: keşif sayfasının yapısı. `public/app.js` sonuç kartlarını üretir.
- `public/history.html`, `public/benchmark.html`: sayfaya özgü yerleşim kuralları ortak renk değişkenlerini kullanır. Geçmiş sayfası yalnızca mevcut sunucu ayarı izin veriyorsa kullanılabilir.

Yeni stiller ortak renkleri tekrar tanımlamaz.

## Renk ve yüzey

Sayfa `#FAFAFA`, kartlar beyaz, yardımcı alanlar `#F2F4F5` olur. `#185C68` ana eylem, marka, bağlantı ve klavye odağında kullanılır. Seçili seçenekler `#EAF3F5` zemin taşır. Ana renk geniş dekoratif alanları kaplamaz.

Metin `#24272B`, yardımcı metin `#616970` olur. Kart sınırı `#E3E6E8`, girdi ve ikincil düğme sınırı `#879399` olur. Başarılı işlem yeşil, uyarı kahverengi, hata kırmızı metinle ve açık zeminle gösterilir. Durumu yalnızca renk anlatmaz; “Saved”, hata açıklaması ve benchmark değerlendirme etiketi görünür kalır.

Oyun kapakları özgün renklerini korur. Başlangıç kartlarının başlıkları görselin altındaki beyaz yüzeyde görünür; görsel üzerinde uzun metin veya koyu geçiş katmanı kullanılmaz. Eksik görsel yerine oyun adı gösterilir.

## Yazı ve dil

Sistem yazı ailesi `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif` olur. Dış font isteği yapılmaz. Gövde 16px/1.65, arayüz 14px/1.5, yardımcı bilgi 13px olur. Başlıklar 600 ağırlık taşır; ana başlık 28–40px, bölüm başlığı 24px ve oyun adı 20px olur.

Arayüzün mevcut dili İngilizcedir; sorgu alanı Türkçe dahil farklı dillerde istek kabul eder. Bu uyarlama yeni bir çeviri sistemi eklemez. Etiketler kullanıcının işlemini anlatır: “Games you love”, “More filters”, “View on Steam”, “Add to taste”. Yeni motivasyon sloganları eklenmez. Uzun oyun adları, sorgular ve düğme etiketleri satıra geçer.

## Yerleşim

Keşif kabuğu en fazla 1440px genişliğindedir, masaüstü kenar boşluğu 40px olur. Sol alan bir gezinme menüsü değil, oyun isteği ve tercihler için 352px arama formudur. Form ile içerik arasında 32px bulunur. 900px ve altında kenar boşluğu 24px, form genişliği 320px olur.

760px ve altında form ve sonuçlar tek sütuna geçer; sayfa kenarı 16px olur. Orta genişliklerde form alanları iki sütun kullanabilir, 600px ve altında yeniden tek sütun olur. Kontroller sayfanın normal akışında kalır; formun bağımsız kaydırma alanı veya sabit yüksekliği yoktur. Masaüstü üst çubuğu sabit kalır, dar ekranda içerik akışına geçer.

Sonuçlar yeterli genişlikte iki sütundur; 1100px altında dar içerik alanında tek sütun olur. Form üstte olduğunda 600–760px aralığında iki sonuç sütunu yeniden kullanılabilir. Oyun kartı iç boşluğu masaüstünde 24px, mobilde 20px olur.

Bir grubun içindeki boşluk 8–12px, gruplar arasında 24px, ana bölümler arasında 32px olur. Metin ve girdi arasında 8px kalır. Hizalamalar ve boşluklar CSS logical properties ile yazılır. Geniş benchmark/geçmiş tabloları kendi alanında kayar; sayfa yatay taşmaz.

## Kontroller ve kartlar

Bir görev grubunda tek ana eylem bulunur: “Find my next game”. Ana düğme petrol mavisi üzerinde beyaz metin kullanır. İkincil düğmeler beyaz zemin ve belirgin kontrol kenarı taşır. Düğmeler ve girdiler en az 44px yüksekliğindedir. Girdi yazısı en az 16px ve köşesi 8px olur; görünür etiket yer tutucu metnin yerini almaz.

Kartlar beyaz zemin, 1px yüzey sınırı ve 12px köşe taşır. Normal kartlarda gölge ve hover hareketi bulunmaz. Hover renk veya kenar değişikliğiyle belirtilir. Oyun ekleme, çıkarma ve geri bildirim kontrolleri satıra geçebilir.

Filtreler, eşleşen metin, ilgili oyunlar ve geri bildirim native `details`/`summary` ile açılır. Açılma işareti görünür kalır. Her etkileşimli öğe klavye odağında 2px petrol mavisi çerçeve gösterir. Seçim radyolarının görünür etiketi odak çerçevesini taşır. Azaltılmış hareket tercihinde animasyon ve geçişler kapanır.

## Durumlar ve veri

Arama sürerken yükleme metni ve nötr kart iskeletleri görünür. Boş sonuç, sorguyu değiştirme veya filtreleri temizleme eylemini sunar. Hata metni sorunu ve mümkün olan sonraki işlemi açıklar. Başarılı oyun seçimi kısa bir durum mesajı gösterir.

Katalog kapsamı ve veri güncelliği açıklaması korunur. Gerçek Steam bağlantısı, oyun görseli, inceleme oranı, oyun modu ve eşleşen kaynak metni kartta kalır. Demo verilerinde mevcut demo etiketi korunur; gerçek veriymiş gibi gösterilmez. Tasarım kuralları veri doğruluğu veya öneri kalitesi garantisi değildir.

## Doğrulama

Yeni görünümler 320, 390, 760, 900, 1280 ve 1440px genişliklerinde kontrol edilir. 200% yakınlaştırma, uzun Türkçe sorgu ve oyun adları, klavye odağı, filtre disclosure, oyun seçimi, sonuç, yükleme ve hata durumları kontrol edilir. RTL yalnızca bir yerleşim dayanıklılık kontrolüdür; yeni dil desteği iddiası değildir. Yapılmayan kontroller raporda açıkça belirtilir.
