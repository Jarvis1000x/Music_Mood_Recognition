# Bi-Modal Music Mood Recognition with Audio and Lyrics

This Repository is an implementation of [Music Mood Detection Based On Audio And Lyrics With Deep Neural Net](http://ismir2018.ircam.fr/doc/pdfs/99_Paper.pdf) by R. Delbouys et al. This model uses two CNN layers and two dense layers to solve Music Emotion Recognition problem. It is using Multi-modal Architecture in Regression task. This bi-modal deep learning structure is expected to combine data from two different domains and reflect information that can not be covered by one domain.

### Dataset

For datasets, the [Deezer Mood Detection Dataset](https://github.com/deezer/deezer_mood_detection_dataset) and parts of the [Million Song Dataset](http://millionsongdataset.com/) was used. The Deezer Mood detection dataset didn't include the audio and the lyrics due to copyright issues and thus had to be supplemented used the Million Song Dataset. However with a few adjustments a dataset can be chosen and used for the task from [this website](http://mir.dei.uc.pt/downloads.html)
