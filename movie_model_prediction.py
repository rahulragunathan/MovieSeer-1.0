#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import numpy as np
import re
from sklearn.externals import joblib
import json
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def GetMoviePrediction(length,cost,release_year,release_month,release_day, \
                       adaptation,actor,cinema,director,distributor, \
                       editor,music,producer,writer,story,studio,plot):
    
    #load movie prediction model
    movie_pred_feat_file = 'models/movie_prediction_features.pkl'
    movie_pred_feat = joblib.load(movie_pred_feat_file)
    no_movie_pred_feat = len(movie_pred_feat)
    actor_prefix = "actor_"
    cinema_prefix = "cinematographer_"
    director_prefix = "director_"
    distributor_prefix = "distributor_"
    editor_prefix = "editor_"
    music_prefix = "music_"
    producer_prefix = "producer_"
    writer_prefix = "writer_"
    story_prefix = "story_"
    studio_prefix = "studio_"

    rnfc_model_file = 'models/rnfc_model.pkl'
    rnfc = joblib.load(rnfc_model_file)
    
    # create test data array
    test_data = np.zeros((1, no_movie_pred_feat))
    
    # cost
    test_data[0,movie_pred_feat.index('costs_clean')] = cost
    
    # length
    test_data[0,movie_pred_feat.index('length_clean')] = length
    
    # release week
    release_week = datetime.date(release_year, release_month, release_day).isocalendar()[1]
    test_data[0,movie_pred_feat.index('release_week')] = release_week
    
    # release day of week
    release_day_of_week = datetime.datetime.weekday(datetime.datetime.strptime(str(release_year)+"-"+str(release_month)+"-"+str(release_day), "%Y-%m-%d"))
    test_data[0,movie_pred_feat.index('release_day_of_week')] = release_day_of_week
    
    # adaptation
    test_data[0,movie_pred_feat.index('adaptation')] = adaptation
    
    # actor
    actor = [re.sub('\[[0-9]+\]',"",item) for item in actor]
    actor = [actor_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in actor]
    for item in actor:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue
            
    # cinema
    cinema = [re.sub('\[[0-9]+\]',"",item) for item in cinema]
    cinema = [cinema_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in cinema]
    for item in cinema:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # director
    director = [re.sub('\[[0-9]+\]',"",item) for item in director]
    director = [director_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in director]
    for item in director:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # distributor
    distributor = [re.sub('\[[0-9]+\]',"",item) for item in distributor]
    distributor = [distributor_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in distributor]
    for item in distributor:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # editor
    editor = [re.sub('\[[0-9]+\]',"",item) for item in editor]
    editor = [editor_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in editor]
    for item in editor:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # music
    music = [re.sub('\[[0-9]+\]',"",item) for item in music]
    music = [music_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in music]
    for item in music:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # producer
    producer = [re.sub('\[[0-9]+\]',"",item) for item in producer]
    producer = [producer_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in producer]
    for item in producer:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # writer
    writer = [re.sub('\[[0-9]+\]',"",item) for item in writer]
    writer = [writer_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in writer]
    for item in writer:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # story
    story = [re.sub('\[[0-9]+\]',"",item) for item in story]
    story = [story_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in story]
    for item in story:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # studio
    studio = [re.sub('\[[0-9]+\]',"",item) for item in studio]
    studio = [studio_prefix + re.sub('[^A-Za-z0-9\s]+',"",item) for item in studio]
    for item in studio:
        try:
            test_data[0,movie_pred_feat.index(item)] = 1.0
        except ValueError:
            continue

    # plot decomposition
    lda_prefix = 'lda_topic_'
    # load plot lda models
#     lda_tf_file = 'models/lda_tf_vectorizer.pkl'
#     lda_tf_vect = joblib.load(lda_tf_file)
#     lda_model_file = 'models/lda_model.pkl'
#     lda_model = joblib.load(lda_model_file)

#     plot_list = [plot]
#     lda_tf = lda_tf_vect.fit_transform(plot_list)
#     lda_W = lda_model.transform(lda_tf)
#     print lda_W
    
#    print test_data
    
    model_pred = rnfc.predict(test_data)
    return model_pred[0]

# define test data
test_length = 109
test_cost = 48000000
test_release_year = 2015
test_release_month = 1
test_release_day = 21
test_adaptation = 0
test_actor = ['Liam Neeson','Forest Whitaker','Famke Janssen','Maggie Grace','Dougray Scott','Sam Spruell','Leland Orser']
test_cine = ['Eric Kress']
test_director = ['Olivier Megaton']
test_distributor = ['20th Century Fox (US)','EuropaCorp (France)']
test_editor = ['Audrey Simonaud','Nicolas Trembasiewicz']
test_music = ['Nathaniel MÌ©chaly']
test_producer = ['Luc Besson']
test_writer = ['Luc Besson','Robert Mark Kamen']
test_story = []
test_studio = ['EuropaCorp','Canal+','CinÌ©+','M6 Films','TSG Entertainment']
test_plot = "In 2014, former covert operative Bryan Mills (Liam Neeson) visits his daughter, Kim (Maggie Grace), to deliver a birthday gift. After an awkward visit, he invites his ex-wife, Lenore (Famke Janssen), to dinner. Although she declines, she shows up at his apartment and tells him about her marital problems. He agrees to let her try to work things out with her current husband Stuart (Dougray Scott). The following day, Bryan receives a text from Lenore asking to meet him for breakfast. Bryan goes out for bagels; and, when he returns to his apartment, he discovers her lifeless body. L.A.P.D. units immediately appear and try to arrest him; but he resists and escapes. Meanwhile, L.A.P.D. Inspector Frank Dotzler (Forest Whitaker) familiarizes himself with Bryan's background and issues a B.O.L.O. for him. Bryan retreats to a safe house equipped with weapons and surveillance electronics. He retraces Lenore's travels to a remote gas station convenience store and finds surveillance footage of her being abducted by unidentified men with unique hand tattoos; but L.A.P.D. detectives arrive and arrest him. While in-transit, Bryan frees himself, hijacks the police cruiser, escapes, and downloads phone records from an L.A.P.D. database onto a thumb drive. He contacts Kim at Lenore's funeral via a camera hidden in his friend Sam's suit, instructing her to maintain her ""very predictable schedule"". She purchases her daily yogurt drink with a ""Drink Me Now"" note which unknowingly is drugged by Bryan. During a lecture, she feels nauseated and runs to the restroom where Bryan is waiting. He surprises her and gives her the antidote to the drug. Bryan removes a surveillance bug that, unknown to her, was planted by Dotzler. He tells her that he is looking for the real murderer and that she should keep safe. Kim tells Bryan of her pregnancy and that Stuart is acting scared and has hired bodyguards which he has never done before. Bryan tails Stuart's car but is ambushed by a pursuing SUV that pushes his car over the edge of a cliff. He survives the crash, hijacks a car, follows the attackers to a roadside liquor store and kills them. Bryan then abducts and interrogates Stuart, who confesses that his failure to repay a debt to a former business partner and ex-Spetsnaz operative named Oleg Malankov (Sam Spruell) was the reason Lenore was killed and that he exposed Bryan's identity to Malankov out of jealousy. With assistance from his old colleagues and a nervous Stuart, Bryan gains entry to Malankov's heavily secured penthouse. After killing the guards, a furious gun battle, and brutal fight, a mortally wounded Malankov reveals that Stuart tricked them both. Stuart had planned Lenore's murder and framed Bryan as part of a business deal to collect on a $12M insurance policy. When Malankov failed to kill Bryan, Stuart used Bryan to kill Malankov and remove all threats. Meanwhile, Stuart shoots Bryan's ally, Sam (Leland Orser), and abducts Kim, intending to flee with the money. Under police pursuit, Bryan arrives at the airport in Malankov's Porsche as Stuart's plane is taxiing toward takeoff. After destroying the landing gear, preventing the plane from taking off, Bryan overpowers Stuart and prepares to kill him but pauses at Kim's pleas. He tells Stuart to expect final punishment if he escapes justice or completes a reduced prison sentence. Dotzler and the LAPD arrive to arrest Stuart. Bryan is acquitted and cleared of all charges. In the aftermath of Stuart's arrest, Kim who is pregnant, informs Bryan that she wants to name her baby after her mother. "

movie_pred = GetMoviePrediction(length=test_length, cost=test_cost, \
                   release_year=test_release_year, release_month=test_release_month, release_day=test_release_day, \
                   adaptation=test_adaptation, actor=test_actor, cinema=test_cine, director=test_director, \
                   distributor=test_distributor, editor=test_editor, music=test_music, producer=test_producer, \
                   writer=test_writer,story=test_story,studio=test_studio,plot=test_plot)
print movie_pred