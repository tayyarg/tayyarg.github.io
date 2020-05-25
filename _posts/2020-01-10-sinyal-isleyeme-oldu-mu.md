---
layout: post
title: Sayısal Sinyal İşleme 1 - Sayısal Sinyal İşleme (DSP) öldü mü?
tags: [dsp, sinyal işleme, 5G, makine öğrenmesi, yapay zeka]
---

Emre: Tüm endüstri Yapay Zeka (AI), 5G, Veri Bilimi ve Makine Öğrenmesi'ne kayıyor gibi. Bu Sayısal Sinyal İşleme (digital signal processing- DSP )'ciler için ne anlama geliyor? DSP öldü mü?

Kaan: Güzel soru. Bu durumda DSP'nin hem bir disiplin olarak hem de bir teknoloji olarak düşünülmesi gerekir. İkisine de bir göz atalım.

Hatta bunu sana ters yönde bir soru sorarak anlatmaya çalışabilirim. 

Sinyal işleme disiplinine neden ihtiyacımız var, yani biz neden sinyalleri işleriz?

Emre: Fiziksel modelleri sinyal olarak modelleyip analiz etmek için. 

Kaan: Aynen. Yani bir nevi fiziksel modeller üzerinden sinyalleri "anlamak" için işleriz değil mi! 

Örneğin bir sinyalin Hızlı Fourier Dönüşümü'nü (FFT) alırız. Bunu sinyalin frekans spektrumu üzerinde çalışmak için yaparız. Bir sinyalin spektrumuna bakarak o sinyalin içinde aradığımız bir örüntü var mı yok mu bunu anlayabiliriz, ya da spektruma dayalı bir filtreleme yapmak istiyor olabiliriz. Ya da elimizdeki sinyali sıkıştırmak isteyebiliriz. Başka bir zaman da sinyal işleme teknikleri kullanarak bazı sinyallerin örnekleme frekansını orjinalinden bir başka frekansa çevirmek isteriz (multi-rate signal processing). Bu liste uzayıp gidebilir. Burada keseceğim. Anlatmak istediğim şey şu; biz sinyalleri çeşitli uzaylarda bileşenlerine ayırarak ya da ayırmadan belirli matematik serileri şeklinde yazarak ya da çeşitli transformasyonlardan geçirerek *anlamaya* çalışırız. 

Peki Makine Öğrenmesi'nin amacı nedir?

Emre: Adından da anlaşıldığı üzere *öğrenmek*. 

Kaan: Evet ama neyi öğrenmek?

Emre: Veri üzerine kurulan modeli.

Kaan: Çok iyi. Yani fiziksel modelleri *anlamaya* dahi gerek duymadan veri üzerine kurulu modeli *öğrenmek*. 

Bu nasıl olabilir diyebilirsin. Ama makine öğrenmesinin güzelliği burada yatıyor. Yüz yıla yakındır geliştirilen binlerce karmaşık sinyal işleme yönteminin analiz ederek anlamaya çalıştığı ve bazen belki de sınıflandıramadığı sinyalleri, bu sinyalleri veri olarak gören derin öğrenme metodları basit bir matematikle öğrenip sınıflanırabiliyor. Bazen ne olduğunu anlamadan (analiz etmeden) öğrenmelerine rağmen sinyal işleme yöntemleri ile ulaştığımızdan daha başarılı sonuçlara ulaşabiliyoruz. Bu bağlamda makine öğrenmesinin sınıflandırma problemlerinde sinyal işlemeden daha etkin olduğunu söyleyebiliriz. Ancak tabiki sınıflandırma problemi toplam problem uzayının küçük bir kısmıdır aslında. Hala daha sadece sinyal işleme yöntemleriyle çözülecek bir çok problem bulunmaktadır.  

Emre: Bu işte bir terslik yok mu? Makine öğrenmesi nasıl anlamadan daha iyi öğrenebiliyor?

Kaan: Derin öğrenmede matematiksel çerçeve en baştan sınıflandırmaya dayalı kurulmuştur. Öğrenme için yapılmış özel bir makine gibi! Sinyal işleme yöntemleri herşeyi doğrusallaştırarak analiz etmeye ya da doğrusal sistemler olarak modellemeye dayalıdır. Arkasında ağır matematik teoremleri vardır. Doğrusal olmayan analizler de yapılır ama işler çok hızlıca çok karmaşık hale gelebilir. Bu nedenle çok sayıda yöntem ve çok detay vardır. Oysa derin öğrenme bu analiz kısmını atlar ve arkasında çok derin teoriler yoktur. Yani anlama işi öğrenme sürecinin bir parçası olmak zorunda değildir. Aslında bakılacak olursa çok katmanlı bir derin öğrenme sisteminin nasıl sınıflandırdığını henüz araştırmacılar da anlayamıyor demek büyük bir adaletsizlik olmaz. 

Emre: Yani iki disipline de farklı problem uzaylarında ihtiyaç duyulmaya devam edecek?

Kaan: Hem evet hem hayır. Bir sınıflanırma probleminin çözümü için sinyalleri öğrenmek yetiyorsa o zaman makine öğrenmesiyle yetinebiliriz. Ancak başka bir problemde sinyali anlamak gerekebilir. O zaman yine sinyal işleme yöntemlerine ihtiyaç duyarız. Böyle durumlarda DSP ile makine öğrenmesi arasındaki ilişkiyi kurmamız da gerekebilir. Örneğin EM (Beklenti-Maksimizasyon) algoritmasının aslında optimale yakın bir maksimum olabilirlik kestirimi olduğunu farketmemiz gerekir. Varsayımımız gürültünün Gauss dağılımı olduğu yönündedir. Peki ya değilse? Yeni bir gürültü dağılımını stokastik olarak nasıl modelleriz? EM'in çalışmadığı yerde ne kullanabiliriz?
Bunun gibi örneklerde, bazen çözümler birden fazla disiplinin birlikte çalışılmasıyla ortaya çıkabilir. Bunun örneklerini ileride inceleyebiliriz.  

Emre: Makine öğrenmesi veya derin öğrenme gibi yeni ortaya çıkan disiplinlerin sinyal işlemeye göre dezavantajı nedir?

Kaan: Teknolojik olarak, makine öğrenmesi algoritmaları, yapay zeka, derin öğrenme ya da veri bilimi genellikle büyük fiziksel sistem belleklerine gereksinim duyarlar. Bu derin öğrenmenin kaçınılmaz olduğu çok büyük bir veri setini işlemek için makul olsa da, sinyal işleme ile basitçe çözülebilecek küçük bir problemi çözebilmek için derin öğrenme ağı kullanmak da bir o kadar gereksiz bellek israfına sebep olacaktır. Neyi nerede kullanmanın uygun olduğunu bilmek biraz ustalık gerektirir.  

Teknoloji demişken, işin başka bir boyutu da dünyanın bu değişimi nasıl karşıladığıdır.

Bunu anlatabilmek için üniversitelerde olup bitene bir göz atıp bir de Texas Instruments (TI)'ın başına gelenlerden bahsetmem gerek. 

Sinyal işlemenin babası sayılan Alan Gatherer'in 2017 yılında IEEE 'de <a href="https://www.comsoc.org/publications/ctn/death-and-possible-rebirth-dsp">DSP'nin ölümü ve muhtemel yeniden doğuşu</a> başlığıyla yayınlanan bir makalesi vardı. Artık okulların sinyal işlemeyi temel bir disiplin olarak görmemesinden, mezunların sinyal işleme konularını eskisi gibi derinlemesine anlamamasından, kampüste ve çevrimiçi bütün okulların ağırlıklı olarak makine öğrenmesi eğitimine kayışından bahsediyor ve bu dönüşümün kabul edilmesi gerektiğini yazıyordu. Eskiden DSP laboratuarları kuran üniversiteler şimdi CUDA ve OpenGL laboratuarları kuruyor ve GPU (grafik işleme üniteleri) programlama dersleri veriyorlar. Sayısal filtre tasarımı, FFT, matris faktorizasyonu, IIR filtreler artık Matlab'ın arka-planda bizim için yaptığı ve sadece özel olarak açılan derslerde anlatılan konular olmaya başladı. 

İşin bir başka boyutu da endüstride olup bitenler.

2014 yılına kadar TI çok-çekirdekli DSP'leri ile işlem kapasitesi bakımından Nvidia ve FPGA'lerle burun farkıyla yarışıyordu. Ama 2010 civarında endüstride yeni bir dalga başlamıştı. Ürün mühendisleri artık "sunucu" / bulut tabanlı çözümlere kayıyordu. Filmi hızlıca günümüze sararsak, istisnasız bütün Yapay Zeka ve 5G uygulamalarından "bulut yerel" (cloud native) uyumluluğu bekleniyor.  

Emre: Bulut Yerel?

Kaan: Yani açık kaynak kodlu, mikroservisler tarafından kurulabilen, "konteyner" (container) içinde çalışan, bulut üzerinden orkestre edilebilen,  optimal kaynak kullanan uygulamalara verilen ad. Bazı 5G ve Yapay Zeka uygulamaları zaman duyarlı verileri işlenmek üzere  coğrafi olarak veri kaynağına yakın ara sunuculara gönderir. Bu sunucularda yine bulut mimarisinin bir parçasıdır. Gerçek zamanlı veri işleme, arabellekleme, bellek tamponu oluşturma, optimizasyon ve M2M gibi "uç" noktasında yapılabilecek işleri yaparlar. Dediğim gibi bu mimari gerçek zamanlı çıkarım (real-time inferencing) ya da gecikme duyarlılığı gibi uygulamalarda ön plana çıkar.

Herneyse, konumuza dönersek. Günümüzde AT&T ve Google gibi büyük oyuncular 5G ve Yapay Zeka uygulamalarının kombinasyonları ve büyük miktarda kablosuz veriyi düşük gecikme ile sunabilmek için yüzlerce hatta binlerce sunucu kuruyor. Bu TI gibi kablosuz baz istasyonu işinde olan firmaların işine gelirdi ancak bu işi sunucu tabanlı olarak yapabiliyor olsalardı. TI uzun yıllardır kullandığı "biz bir elektronik bileşen firmasıyız" mottosundan ve felsefesinden vazgeçmediği için sunuculara giremedi ve oyunun çok gerisinde kaldı. Hala değerlendirme kartları (eval board), JTAG emulator ve IDE geliştirmeye devam ediyor. Bu da TI müşterilerinin zamanla kendini terketmesine sebep oluyor. TI sunucu teknolojilerini kabullenmedikçe yani PCIe kartları, DPDK, VM ve "container" desteği gibi konularda çözümler geliştirmedikçe Yapay Zeka ve 5G dalgasını yakalayamayacak. Bu tür konular Amazon ve Tesla'nın yaptığı gibi biraz daha yeniliği kucaklayıcı yönetim tarzı gerektiriyor. Unutmamak gerekir ki, bugünün sistem mimarisi kararlarını milenyum kuşağı veriyor. Yani donanımdan çok yazılıma dayalı düşünce sistematiği olan insanlar. Bu kuşak kendileriyle birlikte eğitim, endüstri ve paranın da hangi yönde akacağına karar veriyor. Bundan yirmi yıl önce kimin aklına sunucu desteği olan Yapay Zeka çiplerinin firmalara milyar dolarlar kazandırabileceği gelirdi. 

TI benim yakından bildiğim ve ürünleriyle yıllarca geliştirme yaptığım bir firma olduğu için örnek olarak seçtim.

İşin bir başka yönü de, Stanford Üniversite'sinden John Hennessey'in dikkat çektiği nokta. Moore yasası ve Dennard ölçeklemesi bizi yeni bir çağa doğru sürüklüyor. Bu yeni çağda ortaya Domen Spesifik Mimariler (ya da Alana Özelleşmiş Mimariler) çıkabilir. Bu da Alana Özelleşmiş Diller'in çıkması anlamına gelir. Google'un TPU (Tensor Processing Unit)'su buna bir örnek olabilir. Derin Öğrenme'yi iyi düzeyde yapabilmek için TPU mimarisini de iyi düzeyde anlamanız gerekiyor. Yani bir zamanlar (90'larda) iyi DSP yapabilmek için üzerinde çalıştığınız DSP mimarisini iyi anlamanız gerektiği gibi!

Öyleyse özetle makine öğrenmesi, yapay zeka, derin öğrenme ve veri bilimi gibi disiplinlerin sinyal işlemecileri tehdit ettiğini söyleyebiliriz. Bu durumda sinyal işlemeyle uğraşanların bu disiplinlere kayıtsız kalmasını beklemek de güçtür. Neyse ki istatistiksel sinyal işleme konuları makine öğrenmesi konuları ile ciddi manada örtüşür. Bu nedenle gelecekte her ikisini bilen araştırmacılar sadece bir alanda kalanlara göre daha avantajlı bile olabilir. Diğer yandan da artık sadece kendi başına çalışan gömülü sistemler değil bulut ve sunucu tabanlı teknolojiler de DSP mühendislerinin çalışması gereken alanlar arasına girmiş gibi görünüyor. Bu teknolojilere de kayıtsız kalmamaları gerekir. 

Umarım sorunu cevaplayabilmişimdir. 