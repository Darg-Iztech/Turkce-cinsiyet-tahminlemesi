# Türkçe Tweetlerden Cinsiyet Tahminlemesi
Türkçe Toplanmış Tweetler Üzerinden Twitter Kullanıcılarının Cinsiyetlerinin Tahminlenmesi

# Kurulum
- Python 2.7 ya da Python 3.x
- Gerekli Python Kütüphaneleri:
      tensorflow-gpu, gensim, NLTK, numpy, tqdm, matplotlib
- Veri setleri https://pan.webis.de/clef18/pan18-web/author-profiling.html adresinden, ve istenirse daha önceki/sonraki tarihli     yayınlarından da elde edilebilir.
- Char embeddingleri kodlar arasında olup word embeddingleri https://nlp.stanford.edu/projects/glove/ adresinden edinilebilir.
- Çalışmada kullanılan, tarafımızdan toplanmış veri seti için: https://cloud.iyte.edu.tr/index.php/s/5DhqdlUCCdB60qG

# Çalıştırma
- İlgili Parameters dosyasındaki parametreleri ve pathleri kendi kurulumunuza göre ayarladıktan sonra:
    * İsterseniz main.py dosyasındaki kodlara bakarak o dosya ile tek çalıştırma yapabilirsiniz
    * İsterseniz ilgili parametreleri seçtikten sonra eğitim için run.sh kullanarak dilediğiniz accuracy değerinin üstüne çıkan modelleri kaydedebilirsiniz.
    
Kodların çalışmasında gördüğünüz herhangi bir problem için issuelar üzerinden ya da email yoluyla ulaşabilirsiniz.
Eğer araştırmalarınızda bu kodlar veya veri setinden yararlandıysanız, lütfen atıfta bulunarak belirtiniz.
