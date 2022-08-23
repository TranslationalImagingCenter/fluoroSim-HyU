import os
import time
import itertools
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import multiprocessing as mp

import numpy as np
from numpy.random import RandomState
from scipy import stats, ndimage
from fast_histogram import histogram1d
import tifffile as tf


def get_fp_prob_arr(fp_name):
    # matched with literature https://www.fpbase.org/protein/tdtomato/ and doi: 10.1038/nbt1037
    if fp_name == 'tdtomato':
        pdf_val = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00073431, 0.002242  ,
       0.00995257, 0.03185334, 0.07642199, 0.11834476, 0.13041053,
       0.12271254, 0.10743257, 0.09392808, 0.07876141, 0.06149155,
       0.04774304, 0.03425374, 0.0271741 , 0.02056587, 0.01640903,
       0.01091695, 0.00865162])
    # this needs to be fixed
    elif fp_name == 'tdtomato_f':
        pdf_val = np.array([3.69000535e-04, 8.14142160e-04, 1.97841055e-03, 1.53110452e-05,
                            1.86503783e-03, 2.94205013e-05, 4.55487823e-04, 1.84240559e-04,
                            1.25762200e-04, 2.63630057e-05, 8.04715491e-04, 8.19112841e-05,
                            1.70107445e-04, 5.06481184e-04, 1.94372771e-03, 5.35896730e-03,
                            1.40619610e-03, 2.92579322e-02, 1.16434028e-01, 1.33832721e-01,
                            1.29657896e-01, 1.14560266e-01, 1.04682679e-01, 8.79014520e-02,
                            6.86271743e-02, 5.39339952e-02, 4.11350838e-02, 3.26108428e-02,
                            2.51550148e-02, 1.99553047e-02, 1.44398832e-02, 1.16804448e-02])
    # matched with https://www.fpbase.org/protein/citrine/ and doi: 10.1038/nbt945
    elif fp_name == 'citrine':
        pdf_val = np.array([0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0.,
                            0.01448921, 0.05875544, 0.16150872, 0.1957658, 0.15447589,
                            0.10386054, 0.0725873, 0.05952329, 0.04617298, 0.03246164,
                            0.0223158, 0.01712789, 0.01337978, 0.01116128, 0.00733055,
                            0.00679584, 0.00540886, 0.00466423, 0.00384473, 0.00401155,
                            0.00227399, 0.00208469])
    # this needs correction
    elif fp_name == 'citrine_f':
        pdf_val = np.array([5.11398086e-04, 1.21378229e-03, 2.03582338e-03, 9.53813486e-05,
                            1.54629647e-03, 2.24820034e-04, 1.00825761e-03, 6.62343835e-04,
                            8.50856247e-04, 2.07523603e-03, 1.24135099e-02, 6.27105008e-02,
                            1.84157600e-01, 2.14601268e-01, 1.63883719e-01, 8.56944429e-02,
                            4.03990672e-03, 2.23857791e-02, 5.22833424e-02, 4.01458163e-02,
                            3.11019113e-02, 2.48669572e-02, 2.07578744e-02, 1.81083980e-02,
                            1.24473927e-02, 9.91700626e-03, 7.49273230e-03, 6.68060567e-03,
                            5.12480972e-03, 4.26777136e-03, 3.89245786e-03, 2.80200249e-03])
    # corrected and matched with literature https://www.fpbase.org/protein/mko2/
    elif fp_name == 'mko2':
        pdf_val = np.array([0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0.01419252,
                            0.05965109, 0.13508037, 0.16867596, 0.13930274, 0.10692333,
                            0.08604998, 0.0733524, 0.0626534, 0.05148763, 0.03537981,
                            0.02400848, 0.01723534, 0.01331823, 0.00969549, 0.00192643,
                            0.00055004, 0.00051676])
    # this needs correction
    elif fp_name == 'mko2_f':
        pdf_val = np.array([[5.04787952e-04, 1.22386565e-03, 2.08379514e-03, 1.63159756e-04,
                            1.53877583e-03, 2.04873291e-04, 9.70923043e-04, 5.16873666e-04,
                            7.54711745e-04, 7.96585128e-04, 3.25925938e-03, 5.89771668e-03,
                            9.64286584e-03, 1.19283538e-02, 1.75679834e-02, 3.65673236e-02,
                            3.95473106e-03, 6.26524881e-02, 1.71442151e-01, 1.32196880e-01,
                            1.11042448e-01, 1.00460522e-01, 8.71635940e-02, 6.96150514e-02,
                            4.70028163e-02, 3.42737170e-02, 2.48389422e-02, 1.88105110e-02,
                            1.54975513e-02, 1.14272507e-02, 8.81460305e-03, 7.18488998e-03]])
    # corrected matched with https://www.fpbase.org/protein/mruby/
    elif fp_name == 'mruby':
        pdf_val = np.array([0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0.,
                            0., 0.00028933, 0.00236538, 0.01633323, 0.04568238,
                            0.09576676, 0.13509804, 0.14742225, 0.12511571, 0.09532637,
                            0.076426, 0.06321301, 0.05547411, 0.04825485, 0.03938741,
                            0.03041131, 0.02343386])
    # this needs correction
    elif fp_name == 'mruby_f':
        pdf_val = np.array([1.19572645e-03, 2.62499135e-03, 4.02385463e-03, 2.38373885e-04,
                            3.00980391e-03, 4.19365901e-04, 1.80897323e-03, 1.14957606e-03,
                            1.29542734e-03, 1.14518881e-03, 4.12919415e-03, 3.50217955e-03,
                            6.41397524e-03, 5.93080130e-03, 4.70972682e-03, 4.07469033e-03,
                            9.23878270e-04, 3.87332787e-03, 1.81047535e-02, 4.15895544e-02,
                            8.68269968e-02, 1.34709128e-01, 1.50659905e-01, 1.36205447e-01,
                            9.60363753e-02, 7.46471439e-02, 5.81836297e-02, 4.68615111e-02,
                            3.69048581e-02, 2.89308472e-02, 2.23000163e-02, 1.75707791e-02])
    elif fp_name == 'cherry':
        pdf_val = np.array([2.986752783264995514e+01, 6.074481002392585083e+01, 1.167697236484267762e+02,
                            3.518574178313596867e+00, 1.032767351552844417e+02, 1.029140077920808594e+01,
                            4.612711441097505372e+01, 1.397434683681365186e+01, 2.985070611902426663e+01,
                            1.634971147378034573e+01, 6.935690152526481711e+01, 1.109605080331713722e+01,
                            6.322566404114116523e+00, 7.612774718786834427e+00, 1.006922005583250623e+01,
                            2.962000539001164157e+01, 3.019317847662282261e+01, 1.275978056915483876e+02,
                            8.279631643402009331e+02, 2.432553806979075489e+03, 5.658967682758049705e+03,
                            9.283252535357982197e+03, 1.117376729492400591e+04, 9.801439017574830359e+03,
                            7.608099592950579790e+03, 5.954145250364691492e+03, 4.872956090058261907e+03,
                            4.129104795679764720e+03, 3.528898661928462388e+03, 2.862076111625588055e+03,
                            2.192283959406308895e+03, 1.705013744718407679e+03])
        pdf_val = pdf_val / pdf_val.max()
        pdf_val[0:17] = 0.0
        pdf_val = pdf_val / pdf_val.sum()
    # corrected matched with https://pubs.acs.org/doi/pdf/10.1021/ja8045469
    elif fp_name == 'FAD':
        pdf_val = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
                           0.        , 0.        , 0.        , 0.        , 0.        ,
                           0.00264615, 0.01241036, 0.04845612, 0.09025988, 0.11089863,
                           0.11157817, 0.10438517, 0.09438203, 0.08301244, 0.07268156,
                           0.06306063, 0.05054781, 0.04105698, 0.03446981, 0.02619828,
                           0.01894305, 0.01425266, 0.01065471, 0.00794511, 0.00150555,
                           0.00045794, 0.00019696])

    # nothing corrected, matches with https://www.pnas.org/content/pnas/89/4/1271.full.pdf
    elif fp_name == 'NADH_bound':
        pdf_val = np.array([5.39364048e-02, 6.53437219e-02, 7.48992736e-02, 8.07640141e-02,
                            8.29580587e-02, 8.05302811e-02, 7.48066181e-02, 6.75495929e-02,
                            5.98095118e-02, 5.40158914e-02, 4.76339698e-02, 4.21351087e-02,
                            3.58050240e-02, 3.05296847e-02, 2.57019596e-02, 2.14109517e-02,
                            1.78030479e-02, 1.48848783e-02, 1.26060992e-02, 1.08610093e-02,
                            9.22369557e-03, 7.78209448e-03, 6.51414763e-03, 5.49863873e-03,
                            4.54093962e-03, 3.62262545e-03, 3.06080300e-03, 2.64174241e-03,
                            1.89033911e-03, 9.95386039e-04, 1.94439002e-04, 5.00474161e-05])
    # nothing corrected, matches with https://www.pnas.org/content/pnas/89/4/1271.full.pdf
    elif fp_name == 'NADH_free':
        pdf_val = np.array([2.03158770e-02, 3.32399278e-02, 4.66024904e-02, 5.93025762e-02,
                            6.99756411e-02, 7.61207048e-02, 7.76440884e-02, 7.54719678e-02,
                            7.07008378e-02, 6.66967779e-02, 6.09533065e-02, 5.54980027e-02,
                            4.81720852e-02, 4.16076837e-02, 3.53793521e-02, 2.97351134e-02,
                            2.49144822e-02, 2.07471503e-02, 1.74083985e-02, 1.47249059e-02,
                            1.22948671e-02, 1.01315632e-02, 8.17330835e-03, 6.64218350e-03,
                            5.27785289e-03, 4.01934161e-03, 3.19230418e-03, 2.44164776e-03,
                            1.62442635e-03, 7.64932041e-04, 1.53178744e-04, 7.30244688e-05])
    # nothing corrected, matches with https://elifesciences.org/articles/03206 and https://www.pnas.org/content/100/12/7075
    elif fp_name == 'Retinol':
        pdf_val = np.array([0.01628364, 0.02309101, 0.03010651, 0.03713223, 0.04329537,
                            0.04770343, 0.05036413, 0.05229978, 0.05275024, 0.05404962,
                            0.05385399, 0.05398244, 0.05243231, 0.05041978, 0.04728211,
                            0.04367541, 0.03979857, 0.03617128, 0.03261244, 0.02997556,
                            0.02693806, 0.02402532, 0.02097354, 0.01884378, 0.01628235,
                            0.01367686, 0.01144827, 0.00979373, 0.00654174, 0.00334256,
                            0.00060767, 0.00024628])
    # corrected, matched with https://pubs.rsc.org/en/content/articlehtml/2015/ra/c5ra10719a
    elif fp_name == 'Retinoic_acid':
        pdf_val = np.array([0.00323482, 0.00711755, 0.01205687, 0.01798475, 0.02408032,
                           0.02961014, 0.03400473, 0.03800648, 0.04128577, 0.04574504,
                           0.04870653, 0.05146314, 0.05228745, 0.05279178, 0.05195656,
                           0.05052963, 0.04856776, 0.0463301 , 0.04438484, 0.04294912,
                           0.04079648, 0.03821145, 0.03523382, 0.03274007, 0.0292465 ,
                           0.0250353 , 0.02123077, 0.01597456, 0.01179577, 0.00577859,
                           0.00075739, 0.00010592])
    # nothing changed, matches with literature even the bump at 650nm https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-15/issue-04/047008/Changes-of-collagen-and-nicotinamide-adenine-dinucleotide-in-human-cancerous/10.1117/1.3463479.full?SSO=1
    elif fp_name == 'elastin':
        pdf_val = np.array([1.20141868e-01, 1.26277693e-01, 1.24242750e-01, 1.18864403e-01,
                            1.02993134e-01, 8.59644856e-02, 6.64503323e-02, 5.00043599e-02,
                            3.79166758e-02, 2.89040673e-02, 2.46591061e-02, 1.91366906e-02,
                            1.53482380e-02, 1.30254156e-02, 1.02010158e-02, 8.44697626e-03,
                            6.81740579e-03, 5.77340950e-03, 5.03097675e-03, 4.56210305e-03,
                            4.08101315e-03, 3.89418565e-03, 3.49283220e-03, 2.94146614e-03,
                            2.17490939e-03, 1.83347089e-03, 1.61285810e-03, 2.18681653e-03,
                            2.31164190e-03, 6.25552488e-04, 2.55400590e-05, 5.86002637e-05])
    # this is laser, I cannot match this with literature
    elif fp_name == 'bd_unknown':
        pdf_val = np.array([1.68240630e-05, 9.43089753e-05, 1.96854706e-04, 1.36143836e-05,
                            1.91590666e-04, 1.05974539e-04, 1.38758758e-04, 3.71447374e-04,
                            1.95726880e-05, 1.58479699e-03, 4.55848949e-03, 8.00680445e-03,
                            1.06389527e-02, 1.24562655e-02, 1.23719253e-02, 1.07295009e-02,
                            5.77562329e-04, 6.94927099e-03, 1.92964638e-02, 1.79725119e-02,
                            1.60502303e-02, 1.88153163e-02, 2.04985358e-02, 2.87729209e-02,
                            4.20554929e-02, 5.97437973e-02, 7.91393116e-02, 1.03216079e-01,
                            1.26562930e-01, 1.36070531e-01, 1.38618940e-01, 1.24164424e-01])
    elif fp_name == 'laser_refl':
        pdf_val = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.02460835,
                            0.0469063 , 0.10052413, 0.24193936, 0.44226162, 0.12622003,
                            0.01096802, 0.00657219])
    
    elif fp_name == 'gaussian':
        pdf_val = np.array([
            1.489938670, 2.520715180, 4.117703770, 6.494750020,
            9.891128970, 14.54472251, 20.65100664, 28.31087405,
            37.47500016, 47.89678546, 59.10816646, 70.43119185,
            81.03246394, 90.01803543, 96.55538770, 100.0000000,
            100.0000000, 96.55538770, 90.01803543, 81.03246394,
            70.43119185, 59.10816646, 47.89678546, 37.47500016,
            28.31087405, 20.65100664, 14.54472251, 9.891128970,
            6.494750020, 4.117703770, 2.520715180, 1.489938670])
        pdf_val /= pdf_val.sum()

    elif fp_name == 'gaussian1':
        pdf_val = 1/(3 * np.sqrt(2 * np.pi)) * np.exp(-(np.arange(32) - 8)**2 / (2 * 3**2))
        pdf_val /= pdf_val.sum()

    elif fp_name == 'gaussian2':
        pdf_val = 1/(4 * np.sqrt(2 * np.pi)) * np.exp(-(np.arange(32) - 5)**2 / (2 * 4**2))
        pdf_val /= pdf_val.sum()

    #this is CFP
    elif fp_name == 'CFP':
        pdf_val = np.array([0, 0, 0,0,0,0,0.079415623,0.114842485,0.106931858,0.101847327,
                            0.111001224,0.100736074,0.078376708,0.062256616,0.053083896,
                            0.042501629,0.032900535,0.024890429,0.019989385,0.015803256,
                            0.013005152,0.009793386,0.008195066,0.00606786,0.004583819,
                            0.003607781,0.002713123,0.002231124,0.001871264,0.001268629,
                            0.001127967,0.000957786])
    #this is RFP
    elif fp_name == 'RFP':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.014876586,0.03751683,0.0820008,
                            0.121089805,0.12967933,0.124594029,0.103002788,0.094885446,0.078160776,
                            0.060552928,0.045299204,0.033985837,0.024690655,0.019111196,0.013357441,
                            0.009900589,0.007295759])

    # this is YFP, same as citrine
    elif fp_name == 'YFP':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0.015719565,
                            0.061143959,0.158403986,0.185250561,0.150217054,0.104666533,0.080508558,0.065960293,
                            0.051062868,0.036125172,0.026122957,0.017654976,0.013623759,0.009817255,0.00689097,
                            0.005024124,0.003510163,0.002633632,0.002039414,0.001453177,0.001162814,0.001008208])
    #this is GFP
    elif fp_name == 'GFP':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0.162400618,0.228125382,0.171480159,0.115441371,0.089876057,
                            0.071856192,0.05012554,0.032549566,0.023778511,0.015400632,0.011074866,0.00794606,0.005890508,
                            0.003463215,0.002778747,0.002178218,0.001673159,0.001385789,0.00086427,0.000705439,0.000346865,0.000658835])
    #from https://www.thermofisher.com/order/fluorescence-spectraviewer#!/export/
    elif fp_name == 'Alexa405':
        pdf_val = np.array([0.0986977502855000,0.160268293115090,0.157478621038408,0.147222664580154,0.126635797832220,0.0958717047573622,
                            0.0697681379306611,0.0507196474864647,0.0349444897637942,0.0231273684214290,0.0152325197740881,0.00926673262912257,
                            0.00563958185849852,0.00325637043243340,0.00156574287123309,0.000304577223540986,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    #from https://www.thermofisher.com/order/fluorescence-spectraviewer#!/export/
    elif fp_name == 'Alexa488':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0.00193697271804030,0.0128462402371390,0.0565882811542321,0.134502444098201,0.180083669012765,
                            0.163655985908791,0.122473995145730,0.0886887734051406,0.0660112517302702,0.0497986701368752,0.0367636199819243,0.0258322955509210,
                            0.0179126493542452,0.0124507257518178,0.00906873291370883,0.00658423352170390,0.00468220280479867,0.00336737151162604,
                            0.00251122730716989,0.00183033081687,0.00132084023108,0.00108948670695,0,0])

    #from https://www.thermofisher.com/order/fluorescence-spectraviewer#!/export/
    elif fp_name == 'Alexa514':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0.000215516276327171,0.000163281301560211,0.000546008203106511,
                            0.00493338454189579,0.0275848108646285,0.0873209257372162,0.154034651812058,
                            0.166514666574361,0.134315236949943,0.0993216189348570,0.0765182946072034,
                            0.0637108320999084,0.0519461721256971,0.0399442697332940,0.0287325707787641,
                            0.0202088936052416,0.0144329720261332,0.0105528494614755,0.00752902447091649,
                            0.00564769942689273,0.00373726926074257,0.00208905120777759,0,0])


    elif fp_name == 'Alexa546':
        pdf_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00235061940407754, 0.0103878436297784,
                            0.0514059837210175, 0.142114395926549, 0.201929195999036, 0.170708135481761,
                            0.111632739546263, 0.0734480901262019, 0.0531811230901254, 0.0464338811279332,
                            0.0407082393541393, 0.0317718425922824, 0.0207662571661037, 0.0146671352285578,
                            0.0102929671970383, 0.00748550317993251, 0.00601612903799025, 0.00469991819121285])



    elif fp_name == 'Alexa594':
        pdf_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0105726614170125,
                            0.0328340097404750, 0.0770435629030373, 0.129904853047777, 0.158147290202084,
                            0.148383139165347, 0.118890437232037, 0.0882135213449553, 0.0668367966946166,
                            0.0542563775915059, 0.04529217367345, 0.03835561415545, 0.0312695628322525])


    elif fp_name == 'Alexa610':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00156141142132084,
                            0.00375189385614650,0.0162207235673909,0.0555928527329342,0.132517909538114,
                            0.206600606961241,0.200029477843290,0.141212185294981,0.0879894586617234,
                            0.0547167919945871,0.0395294369060703,0.0322055443286248,0.0280717068935759])


    elif fp_name == 'Alexa633':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0139485683644806,
                            0.0497239398447423,0.121202604287472,0.188342966522197,0.198018976553276,
                            0.163445593023946,0.117330122621764,0.0831473881219526,0.0648398406601694])


    elif fp_name == 'Alexa647':
        pdf_val = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0111049702605481,0.0409566117188420,
                            0.111752766742095,0.209755596930076,0.260401425556358,0.217154075751321,0.14887455304076])


    elif fp_name == 'DAPI':
        pdf_val = np.array([0.0264852930338649,    0.0430788985829465, 0.0594673746647101,
                            0.0742044902939655, 0.0836364842908985,    0.0875832812365844,    0.0884593221384643,
                            0.0835418602814561,    0.0767166394314154,    0.0683072086505058,    0.0591590778194846,
                            0.0509755234203919,    0.0426027210024145,    0.0352036410290793,    0.0286744957533995,
                            0.0235232256527733,    0.0188884128107644,    0.0151400364839233,    0.0118776041194074,
                            0.00958950570619777,    0.00728064797110770,    0.00560425562624483,    0,    0, 0,    0,
                            0,    0,    0,    0,     0,    0])

    return pdf_val


def convert_photoncounts_to_digitallevels_LSM780(sim_spectrum):
    # Photon conversion (S factor) from dark calibration, fit results:
    # Sigma_0 = 5.043589e+01
    # Offset = -3.561094e+01
    # A factor = 18
    # S_factor = 2.014142e+03

    # S_factor = np.squeeze(conversation_rate_for_gain)*interp_sensitivity/(1000*(interp_QE*0.01))
    # todo check the s_factor for different gain
    S_factor = 430#2014/100
    measured_sim_spectrum = np.multiply(S_factor, sim_spectrum)
    
    return measured_sim_spectrum


def simulate_spectral_photon_emission_noise_Ncombinations(photon_number, bkg_noise, combo_spectra, combo_ratios, fp_list, add_noise=0):
    """This function simulates fluorescence emission of photons using spectrum as a probability density function
    (with integral normalized to 1). Photons are generated according to this distribution, simulating the real
    measurement we obtain. Photon counts-to-analog Scaling will be proportional to gain and sensitivity of the
    detector, ultimately should be a constant number. (Requires calibration for absolute value calibration.
    PDF here are based on values obtained from Zeiss 780 quasar detector
    
    For combination: the order has some importance, fluorescence spectra pdf are associated a number (1,n).
    Check combination, pick spectra and combine them.
    combo_spectra is an array containing the spectra indexes that will be combined
    combo_ratios is an array with ratios of various spectra
    It is expected that len(combo_spectra) == len(combo_ratios)"""
    
    spectra_matrix = np.zeros([1,32])
    ground_true_spectrum = np.zeros([1,32])
    ground_spectra = np.zeros([1,32])
    for i, ind in enumerate(combo_spectra):
        # matched with literature https://www.fpbase.org/protein/tdtomato/ and doi: 10.1038/nbt1037
        fp_name = fp_list[ind]
        pdf_val = get_fp_prob_arr(fp_name)
        
        channel_num = np.arange(32)
        
        # this does not work if size = 0
        
        if combo_ratios[i] > 0:
            sim_photons = np.random.choice(channel_num, size=np.int(photon_number*combo_ratios[i]), replace=True,
                                           p=pdf_val)  # this is the channel where the photon ends
            sim_spectrum, _ = np.histogram(sim_photons, 32, [0, 31])
            
            ground_spectra += pdf_val*combo_ratios[i]
            # convert photon counts to digital levels for LSM780
            # spectra_matrix +=sim_spectrum*(2014/10)
            spectra_matrix = np.add(spectra_matrix,convert_photoncounts_to_digitallevels_LSM780(sim_spectrum),out=spectra_matrix,casting="unsafe")
            # spectra_matrix = np.sum(spectra_matrix,convert_photoncounts_to_digitallevels_LSM780(sim_spectrum))
            # spectra_matrix += convert_photoncounts_to_digitallevels_LSM780(var(sim_spectrum))
    
    # this adds poisson noise if selected
    if add_noise:
        spectra_matrix = np.random.poisson(spectra_matrix)
    
    # pick a random background noise value
    for j in range(0, spectra_matrix.shape[0]):
        bkg_noise_values = bkg_noise[np.random.randint(bkg_noise.shape[0]), :]
    
    ground_true_spectrum = (spectra_matrix.max()/ground_spectra.max())*ground_spectra
    ground_true_spectrum = np.int16(ground_true_spectrum)
    
    sim_spectrum_final = spectra_matrix + bkg_noise_values
    sim_spectrum_final = np.uint16(sim_spectrum_final)
    
    # TODO: this is just the photons, need to make histogram. Was testing histogramdd for making a larger number of spectra at the same time. https://docs.scipy.org/doc/np-1.15.0/reference/generated/np.histogramdd.html
    
    return sim_spectrum_final, ground_true_spectrum# measured_sim_spectrum, ground_true_spectrum


def simulate_spectral_photon_emission_noise_Ncombinations_average(photon_number, bkg_noise, combo_spectra, combo_ratios, fp_list, add_noise=0, avg_frame=1):
    sim_spectrum_final = np.zeros((1, 32), dtype="float")
    ground_true_final = np.zeros((1, 32), dtype="float")
    
    for f in range(avg_frame):
        sim_spectrum, ground_true_spectrum = simulate_spectral_photon_emission_noise_Ncombinations(photon_number, bkg_noise, combo_spectra, combo_ratios, fp_list, add_noise)
        sim_spectrum_final += sim_spectrum.astype("float")
        ground_true_final += ground_true_spectrum.astype("float")
    
    sim_spectrum_final /= avg_frame
    ground_true_final /= avg_frame
    
    return sim_spectrum_final.astype("uint16"), ground_true_final.astype("uint16")


def load_background_spectra_shared(filepath):
    back_spectra = load_background_spectra(filepath)

    t1_1 = time.perf_counter()
    back_size = np.asscalar(np.prod(back_spectra.shape))
    back_ctype = np.ctypeslib.as_ctypes_type(back_spectra.dtype)
    back_share = mp.RawArray(back_ctype, back_size)
    back_arr = np.ctypeslib.as_array(back_share)
    back_arr[:] = back_spectra.ravel()[:]
    t1_2 = time.perf_counter()
    
    print("shared memory calibration: " + str(t1_2 - t1_1))
    
    return back_share, back_spectra


def create_optical_filter_array():
    fil = np.ones(32)
    # 488 filter
    fil[7:10] = [0.8175, 0.06, 0.3447]
    # 561 filter
    fil[15:19] = [0.64, 0, 0.4, 1]

    return fil


def save_input_spectra(spectratype_list, output_dir, filename, filter_string='filter'):
    spectra_fn = filename + "-spectra"
    spectra_filter_fn = filename + "-spectra" + filter_string

    spectra_fp = os.path.join(output_dir, spectra_fn + ".npy")
    spectra_filter_fp = os.path.join(output_dir, spectra_filter_fn + ".npy")

    # # Generate spectra array
    spectra_shape = get_fp_prob_arr(spectratype_list[0]).shape
    spectra_arr = np.zeros((len(spectratype_list),) + spectra_shape)
    for n in range(len(spectratype_list)):
        spectra_arr[n, :] = get_fp_prob_arr(spectratype_list[n])
    fil = create_optical_filter_array()
    spectra_filter_arr = spectra_arr * fil[np.newaxis, :]

    np.save(spectra_fp, spectra_arr)
    np.save(spectra_filter_fp, spectra_filter_arr)


def save_threshold_list(threshold_list, output_dir, filename):
    thresh_fn = filename + "_threshold_list"

    thresh_fp = os.path.join(output_dir, thresh_fn + ".txt")

    with open(thresh_fp, 'w') as file:
        thresh_iter = iter(threshold_list)
        file.write(str(next(thresh_iter)))
        for el in thresh_iter:
            file.write(', ' + str(el))


def check_input_images(mask_dir, file_type, spectratype_list, dim_str):
    # Check image file information (making sure everything is consistent)
    filepath_list = [os.path.join(mask_dir, fn) for fn in os.listdir(mask_dir) if fn.endswith(file_type)]
    if len(spectratype_list) > len(filepath_list):
        raise AssertionError("Number of FP_list (" + str(len(spectratype_list)) + ") is greater than the " +
                             "number of files (" + str(len(filepath_list)) + ").")
    filepath_list = filepath_list[0:len(spectratype_list)]

    initial_shape_list = []
    dim_order_list = []
    final_shape_list = []
    for filepath in filepath_list:
        img_raw = tf.TiffFile(filepath)
        img_axes_str = img_raw.series[0].axes

        if len(img_axes_str) > len(dim_str):
            raise AssertionError("Axes (" + img_axes_str + ") length is greater than " + str(len(dim_str)))
        dim_bool = True
        for str_el in img_axes_str:
            dim_bool = dim_bool and (str_el in img_axes_str)
        if not dim_bool:
            raise AssertionError("Axes (" + img_axes_str + ") element is not in " + dim_str)

        nd_shape = (1,) * (len(dim_str) - len(img_axes_str)) + img_raw.series[0].shape
        nd_axes_str = ''.join([str_el for str_el in dim_str if str_el not in img_axes_str]) + img_axes_str
        dim_order = tuple([nd_axes_str.index(s) for s in dim_str])

        initial_shape_list.append(nd_shape)
        dim_order_list.append(dim_order)
        final_shape_list.append(tuple((np.array(nd_shape)[np.array(dim_order)]).tolist()))

    # Check if all array shapes match
    bool_shape = True
    for list_ind in range(1, len(initial_shape_list)):
        bool_shape = bool_shape and (final_shape_list[0] == final_shape_list[list_ind])

    if not bool_shape:
        raise AssertionError("Not all image array shapes match.")

    return filepath_list, initial_shape_list, dim_order_list


def check_photon_mask(photon_full_arr, ph_avg):
    photon_full_arr_smooth = np.copy(photon_full_arr)
    Gauss2DKern_mat = gkern(10, 4)
    PSF_rounds = 2
    mask_pnoise = 1

    for img_ind in range(photon_full_arr.shape[0]):
        for i in range(PSF_rounds):
            photon_full_arr_smooth[img_ind, 0, 0, :, :] = ndimage.convolve(photon_full_arr_smooth[img_ind, 0, 0, :, :],
                                                                           Gauss2DKern_mat, mode='reflect')
        photon_full_arr_poiss = np.copy(photon_full_arr)
        rng = RandomState(4213)
        for i in range(mask_pnoise):
            photon_full_arr_poiss[0, 0, 0, :, :] = rng.poisson(photon_full_arr_poiss[0, 0, 0, :, :]).astype("float")

    photon_full_arr_poiss[photon_full_arr_poiss > ph_avg] = ph_avg
    print("finish")
    return photon_full_arr_smooth, photon_full_arr_poiss


def generate_photon_mask(filepath_list, initial_shape_list, dim_order_list, dim_str,
                         intensity_threshold_list, photon_avg, photon_threshold, s_factor_value):
    files_num = len(filepath_list)
    # Create photon masks
    img_arr_list = []
    for f_ind, filepath in enumerate(filepath_list):
        img_raw = tf.TiffFile(filepath)

        nd_shape = initial_shape_list[f_ind]
        dim_order = dim_order_list[f_ind]

        img_arr = img_raw.asarray()
        img_arr = img_arr.reshape(nd_shape)
        img_arr = img_arr.transpose(dim_order)
        img_arr_list.append(img_arr)

    channel_index = dim_str.index('C')  # Determine which index is the channel dimension for averaging

    photon_arr_shape = list(img_arr_list[0].shape)
    photon_arr_shape.pop(channel_index)
    photon_full_arr_shape = (len(img_arr_list),) + tuple(
        photon_arr_shape)  # shape of photon array for all spectra types

    # Set up multiprocessing shared array for full photon array
    ph_size = np.prod(photon_full_arr_shape).item()
    uint16_ctype = np.ctypeslib.as_ctypes_type(np.uint16)
    photon_full_arr_share = mp.RawArray(uint16_ctype, ph_size)
    photon_full_arr = np.ctypeslib.as_array(photon_full_arr_share).reshape(photon_full_arr_shape)

    # Calculate photon values and input into photon array
    for img_ind, img_arr_init in enumerate(img_arr_list):
        img_arr = np.copy(img_arr_init)
        if img_arr.shape[channel_index] == 1:
            img_arr[img_arr < intensity_threshold_list[img_ind]] = 0
            photon_arr = img_arr.astype("float")
            photon_arr_min = photon_arr[photon_arr > 0].min()
            photon_arr_mean = photon_arr[photon_arr > 0].mean()
            photon_arr_max = photon_arr.max()
            photon_arr_std = photon_arr.std()
            photon_arr *= photon_avg / photon_arr_max
            print("Ind: " + str(img_ind))
            print("   photon mean: " + str(photon_arr_mean))
            print("   photon min: " + str(photon_arr_min))
            print("   photon max: " + str(photon_arr_max))
            print("   photon std: " + str(photon_arr_std))


            photon_arr[photon_arr < photon_threshold] = 0

            # #Pimp my mask
            # if img_ind==0:
            #     photon_arr[photon_arr>0]+=1
            #     photon_arr[photon_arr>photon_avg]= photon_avg
            # if img_ind == 1:
            #     photon_arr[photon_arr > 0] += 1
            #     photon_arr[photon_arr > photon_avg] = photon_avg
            # if img_ind==3:
            #     photon_arr[photon_arr>0]+=2
            #     photon_arr[photon_arr>photon_avg]= photon_avg
            photon_arr = np.ceil(photon_arr)
            print("After:")
            photon_arr_min = photon_arr[photon_arr > 0].min()
            photon_arr_mean = photon_arr[photon_arr > 0].mean()
            photon_arr_max = photon_arr.max()
            photon_arr_std = photon_arr.std()
            print("   photon mean: " + str(photon_arr_mean))
            print("   photon min: " + str(photon_arr_min))
            print("   photon max: " + str(photon_arr_max))
            print("   photon std: " + str(photon_arr_std))


        else:
            photon_arr = img_arr.sum(axis=channel_index, dtype="float")

            photon_arr /= s_factor_value * files_num  # go from intensity to photons, then "average" to prevent overflow
            photon_arr *= photon_avg / photon_arr.max()
            # photon_arr -= photon_background  # "remove" background noise
            photon_arr[photon_arr < photon_threshold] = 0  # make sure there aren't too few photons

        photon_arr = photon_arr.astype("uint16")  # store as uint16 for smaller memory footprint
        photon_full_arr[img_ind, ...] = photon_arr

    check_photon_mask(photon_full_arr, photon_avg)
    print("finish")
    return photon_full_arr, img_arr_list, channel_index, photon_full_arr_share, photon_full_arr_shape


def init_simshape_pool(photon_full_share, stochastic_spectra_share_save, true_spectra_share_save,
                       photon_full_arr_shape, spectra_array_process_shape, save_array_shape):
    global ph_values_share, stoch_save, trspec_save, ph_arr_shape, spectra_arr_shape, save_arr_shape
    ph_values_share = photon_full_share
    stoch_save = stochastic_spectra_share_save
    trspec_save = true_spectra_share_save
    ph_arr_shape = photon_full_arr_shape
    spectra_arr_shape = spectra_array_process_shape
    save_arr_shape = save_array_shape


def simulate_shape_spectra_multi_implem(ph_ind_arr, spectratype_list, s_factor_value, seed, number_average_frames,
                                        add_noise=0):
    # global bkg_values_share, ph_values_share, spec_share, tr_share, bkg_save, stoch_save, after_po_save, trspec_save,\
    #     bkg_arr_shape, ph_arr_shape, spectra_arr_shape, save_arr_shape
    global ph_values_share, stoch_save, trspec_save, ph_arr_shape, spectra_arr_shape, save_arr_shape

    ph_2d_shape = (ph_arr_shape[0], np.prod(ph_arr_shape[1:]).item())
    photon_full_2d_arr = np.ctypeslib.as_array(ph_values_share).reshape(ph_2d_shape)

    # bkg_values_arr = np.ctypeslib.as_array(bkg_values_share).reshape(bkg_arr_shape)

    # spectra_2d_shape = (np.prod(spectra_arr_shape[:-1]).item(), spectra_arr_shape[-1])
    save_3d_shape = (save_arr_shape[0], np.prod(save_arr_shape[1:-1]).item(), save_arr_shape[-1])
    # spectra_arr = np.ctypeslib.as_array(spec_share).reshape(spectra_2d_shape)
    # true_arr = np.ctypeslib.as_array(tr_share).reshape(spectra_2d_shape)
    # back_spectra_arr_save = np.ctypeslib.as_array(bkg_save).reshape(save_3d_shape)
    stochastic_spectra_arr_save = np.ctypeslib.as_array(stoch_save).reshape(save_3d_shape)
    true_spectra_arr_save = np.ctypeslib.as_array(trspec_save).reshape(save_3d_shape)
    # after_poisson_arr_save = np.ctypeslib.as_array(after_po_save).reshape(save_3d_shape)

    num_channels = spectra_arr_shape[-1]
    # sim_spectrum_avg_instance = np.zeros((1, num_channels), dtype="float")
    # ground_true_avg_instance = np.zeros((1, num_channels), dtype="float")
    sim_spectrum_instance = np.zeros((1, num_channels), dtype="float")
    # ground_true_instance = np.zeros((1, num_channels), dtype="float")
    pdf_val_arr = np.zeros((len(spectratype_list), num_channels))
    # print("spectratype_list: " + str(spectratype_list))
    for s in range(len(spectratype_list)):
        pdf_val_arr[s, :] = get_fp_prob_arr(spectratype_list[s])
    channel_arr = np.arange(num_channels)

    rng = RandomState(seed)
    for ph_index in range(ph_ind_arr[0], ph_ind_arr[1]):
        # sim_spectrum_avg_instance *= 0
        # ground_true_avg_instance *= 0
        photon_ph_index_vec = photon_full_2d_arr[:, ph_index]
        ground_true_instance = (photon_ph_index_vec[:, np.newaxis] * pdf_val_arr * s_factor_value).sum(axis=0)
        true_spectra_arr_save[:, ph_index, :] = ground_true_instance[np.newaxis, np.newaxis, :]
        for avg_ind in range(number_average_frames):
            sim_spectrum_instance *= 0
            # ground_true_instance *= 0

            for spectype_ind in range(ph_2d_shape[0]):
                photon_val = photon_ph_index_vec[spectype_ind]
                if photon_val > 0:
                    pdf_val = pdf_val_arr[spectype_ind, :]
                    sim_photons = rng.choice(channel_arr, size=np.int(photon_val),
                                             replace=True, p=pdf_val)
                    sim_spectrum = histogram1d(sim_photons, channel_arr.size, [channel_arr[0], channel_arr.size])
                    np.add(sim_spectrum_instance, sim_spectrum, out=sim_spectrum_instance)
                    # np.add(ground_true_instance, pdf_val*photon_val*s_factor_value, out=ground_true_instance)

            stochastic_spectra_arr_save[avg_ind, ph_index, :] = sim_spectrum_instance
            # true_spectra_arr_save[avg_ind, ph_index, :] = ground_true_instance


def simulate_shape_spectra(photon_full_arr_share, photon_full_arr_shape, spectra_array_process_shape, save_array_shape,
                           spectratype_list, s_factor_value, number_average_frames, add_noise):
    spec_arr_tot_size = np.prod(spectra_array_process_shape).item()
    save_arr_tot_size = np.prod(save_array_shape).item()
    uint16_ctype = np.ctypeslib.as_ctypes_type(np.uint16)
    float_ctype = np.ctypeslib.as_ctypes_type(np.float)
    # spectra_shared = mp.RawArray(uint16_ctype, spec_arr_tot_size)
    # spectrum_arr = np.ctypeslib.as_array(spectra_shared).reshape(spectra_array_process_shape)
    # true_shared = mp.RawArray(uint16_ctype, spec_arr_tot_size)
    # true_arr = np.ctypeslib.as_array(true_shared).reshape(spectra_array_process_shape)
    # back_spectra_share_save = mp.RawArray(uint16_ctype, save_arr_tot_size)
    # back_spectra_arr_save = np.ctypeslib.as_array(back_spectra_share_save).reshape(save_array_shape)
    stochastic_spectra_share_save = mp.RawArray(uint16_ctype, save_arr_tot_size)
    stochastic_spectra_arr_save = np.ctypeslib.as_array(stochastic_spectra_share_save).reshape(save_array_shape)
    # after_poisson_share_save = mp.RawArray(uint16_ctype, save_arr_tot_size)
    # after_poisson_arr_save = np.ctypeslib.as_array(after_poisson_share_save).reshape(save_array_shape)
    true_spectra_share_save = mp.RawArray(float_ctype, save_arr_tot_size)
    true_spectra_arr_save = np.ctypeslib.as_array(true_spectra_share_save).reshape(save_array_shape)

    cores = int(mp.cpu_count() / 2)
    # init_simshape_pool(back_spectra_share, photon_full_arr_share, spectra_shared, true_shared,
    #                    back_spectra_share_save, stochastic_spectra_share_save, after_poisson_share_save,
    #                    bkg_spectra_shape, photon_full_arr_shape, spectra_array_process_shape,
    #                    save_array_shape)
    # pool_multicore = mp.Pool(processes=cores, initializer=init_simshape_pool,
    #                          initargs=(back_spectra_share, photon_full_arr_share, spectra_shared, true_shared,
    #                                    back_spectra_share_save, stochastic_spectra_share_save, after_poisson_share_save,
    #                                    true_spectra_share_save,
    #                                    bkg_spectra_shape, photon_full_arr_shape, spectra_array_process_shape,
    #                                    save_array_shape))
    pool_multicore = mp.Pool(processes=cores, initializer=init_simshape_pool,
                             initargs=(photon_full_arr_share, stochastic_spectra_share_save, true_spectra_share_save,
                                       photon_full_arr_shape, spectra_array_process_shape, save_array_shape))

    seeds = np.random.randint(65535, size=cores)

    n_specsamples_size = np.prod(spectra_array_process_shape[:-1]).item()
    ph_inds = np.floor(np.linspace(0, n_specsamples_size, cores + 1)).astype("int")
    ph_inds = list(zip(ph_inds[:-1], ph_inds[1:]))

    results = []
    print("running pools")
    for c in range(cores):
        # print("setting up core: " + str(c))
        fnc_args = (ph_inds[c], spectratype_list, s_factor_value, seeds[c], number_average_frames, add_noise)
        # simulate_shape_spectra_multi_implem(*fnc_args)
        results.append(pool_multicore.apply_async(simulate_shape_spectra_multi_implem, args=fnc_args))
    pool_multicore.close()

    print("getting results")
    for p in results:
        p.get()
    pool_multicore.terminate()

    return stochastic_spectra_arr_save, true_spectra_arr_save


def get_shapesim_save_fn(dirname, filename, fn_cut):
    """
    Getting filenames for tiff or npy in a new folder at the same level as the input directory
    """
    previous_dir, _ = os.path.split(dirname)
    new_dir = os.path.join(previous_dir, filename)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
    filename_list = ['simulated', 'simulatedfilter', 'true', 'truefilter', 'ratio', 'photons',
                     "background", "stochastic", "poisson"]
    ext_list = ['.tif'] * len(filename_list)
    ext_list[filename_list.index('ratio')] = '.npy'
    filext_list = list(zip(filename_list, ext_list))
    filename_list = [os.path.join(new_dir, filename[:fn_cut] + '-' + fn + ext) for fn, ext in filext_list]
    return filename_list


def addsalt_pepper(img, SNR):
    img_ = img.copy()
    mask = np.random.choice((0, 1, 2, 3, 4), size=img.shape,
                            p=[SNR, (1 - SNR) / 2., (1 - SNR) / 4., (1 - SNR) / 8., (1 - SNR) / 8.])
    img_[mask == 1] = 0  # pepper noise
    img_[mask == 2] = np.random.randint(0, img.mean() * 1 / 30, np.count_nonzero(mask == 2))  # salt noise
    img_[mask == 3] = np.random.randint(0, img.mean() * 1 / 20, np.count_nonzero(mask == 3))  # salt noise
    img_[mask == 4] = np.random.randint(0, img.mean() * 1 / 15, np.count_nonzero(mask == 4))  # hot pixels
    return img_


def generate_photon_mask_simulation(filename, mask_dir, background_path, file_type, spectratype_list,
                                    photon_avg, photon_threshold, intensity_threshold_list,
                                    s_factor_value, number_average_frames,
                                    poisson_iteration, psf_iteration, back_spectra_number, sigtonoise_ratio):
    # Define file and dimension parameters
    dim_str = 'TZCYX'
    dim_use_str = 'TZYXC'
    # file_type = 'tif'
    # photon_background = 10
    # photon_threshold = 10
    # photon_max = 50

    # Check image file information and get input information
    filepath_list, initial_shape_list, dim_order_list = \
        check_input_images(mask_dir, file_type, spectratype_list, dim_str)

    fn_cut = 12
    # Get output filenames
    sim_fn, simfil_fn, true_fn, truefil_fn, ratio_fn, photon_fn, back_fn, stochastic_fn, poisson_fn = \
        get_shapesim_save_fn(mask_dir, filename, fn_cut)

    out_dir, _ = os.path.split(sim_fn)
    cut_filename = filename[:fn_cut]
    # Save input spectra
    save_input_spectra(spectratype_list, out_dir, cut_filename)

    # Save threshold list
    save_threshold_list(intensity_threshold_list, out_dir, cut_filename)

    # Create photon masks
    photon_full_arr, img_arr_list, channel_index, photon_full_arr_share, photon_full_arr_shape = \
        generate_photon_mask(filepath_list, initial_shape_list, dim_order_list, dim_str,
                             intensity_threshold_list, photon_avg, photon_threshold, s_factor_value)

    tf.imsave(photon_fn, photon_full_arr.transpose((1, 2, 0, 3, 4)))

    # Generate ratio array
    ratio_full_arr = photon_full_arr.astype("float")
    ratio_full_arr /= ratio_full_arr.sum(axis=0)
    ratio_full_arr[np.isnan(ratio_full_arr)] = 0
    np.save(ratio_fn, ratio_full_arr)

    # Load background spectra
    print("loading background")
    # out_dir, _ = os.path.split(background_path)
    # back_spectra_share, bkg_spectra = load_background_spectra_shared(background_path)

    # Create shape used during spectra construction and order to get that shape (t,z,c,y,x) > (t,z,y,x,c)
    spec_arr_process_shape = img_arr_list[0].shape  # (t, z, c, y, x)
    if spec_arr_process_shape[channel_index] != 32:
        spec_arr_process_shape = spec_arr_process_shape[:channel_index] + (32,) + spec_arr_process_shape[
                                                                                  channel_index + 1:]
    dim_process_order = [dim_str.index(el) for el in dim_use_str]
    spec_arr_process_shape = tuple((np.array(spec_arr_process_shape)[np.array(dim_process_order)]).tolist())
    save_arr_shape = (number_average_frames,) + spec_arr_process_shape

    print("starting spectra simulation")

    # Daniel: here I create the photon stochastic mask, the background and the ground truth. I removed poisson noise and conversion with s_factor as this happens after the PSF
    # spectrum_arr, true_arr, background_arr, stochastic_arr, poisson_arr, true_spectra_save_arr = \
    #     simulate_shape_spectra(back_spectra_share, photon_full_arr_share,
    #                            bkg_spectra.shape, photon_full_arr_shape, spec_arr_process_shape, save_arr_shape,
    #                            spectratype_list, s_factor_value, 1, poisson_iteration)  #averaging should be done at image level at the end
    stochastic_arr, true_spectra_save_arr = \
        simulate_shape_spectra(photon_full_arr_share, photon_full_arr_shape, spec_arr_process_shape, save_arr_shape,
                               spectratype_list, s_factor_value, number_average_frames,
                               poisson_iteration)  # averaging should be done at image level at the end

    # Daniel: here I use photon stochastic mask, the background and the ground truth in the form of images and add the noise
    # add PSF, poisson and readout noise
    spectrum_arr, true_arr, spectra_filter, true_spectrum_filter, background_arr, stochastic_arr_smooth, poisson_arr = \
        image_PSF_poisson_readout_noise(true_spectra_save_arr, stochastic_arr, photon_full_arr, photon_avg,
                                        s_factor_value, poisson_iteration, psf_iteration, back_spectra_number,
                                        sigtonoise_ratio)

    dim_reverse_order = [dim_use_str.index(el) for el in dim_str]
    spectrum_arr = spectrum_arr.transpose(dim_reverse_order)
    true_arr = true_arr.transpose(dim_reverse_order)
    spectra_filter = spectra_filter.transpose(dim_reverse_order)
    true_spectrum_filter = true_spectrum_filter.transpose(dim_reverse_order)

    import matplotlib.pyplot as plt
    aa = spectrum_arr[0, 0, 15, :, :]

    tf.imsave(sim_fn, spectrum_arr)
    tf.imsave(true_fn, true_arr)

    tf.imsave(simfil_fn, spectra_filter)
    tf.imsave(truefil_fn, true_spectrum_filter)

    for avg_ind in range(number_average_frames):
        background_arr_instance = background_arr[avg_ind, ...].transpose(dim_reverse_order)
        back_fn_instance, back_fn_ext = os.path.splitext(back_fn)
        back_fn_instance = back_fn_instance + "_avgind_" + str(avg_ind) + back_fn_ext
        tf.imsave(back_fn_instance, background_arr_instance)

        stochastic_arr_instance = stochastic_arr[avg_ind, ...].transpose(dim_reverse_order)
        stoch_fn_instance, stoch_fn_ext = os.path.splitext(stochastic_fn)
        stoch_fn_instance = stoch_fn_instance + "_avgind_" + str(avg_ind) + stoch_fn_ext
        tf.imsave(stoch_fn_instance, stochastic_arr_instance)

        stochastic_arr_smooth_instance = stochastic_arr_smooth[avg_ind, ...].transpose(dim_reverse_order)
        stoch_sm_fn_instance, stoch_sm_fn_ext = os.path.splitext(stochastic_fn)
        stoch_sm_fn_instance = stoch_sm_fn_instance + "_smooth_avgind_" + str(avg_ind) + stoch_sm_fn_ext
        tf.imsave(stoch_sm_fn_instance, stochastic_arr_smooth_instance)

        poisson_arr_instance = poisson_arr[avg_ind, ...].transpose(dim_reverse_order)
        poiss_fn_instance, poiss_fn_ext = os.path.splitext(poisson_fn)
        poiss_fn_instance = poiss_fn_instance + "_avgind_" + str(avg_ind) + poiss_fn_ext
        tf.imsave(poiss_fn_instance, poisson_arr_instance)


def background_spectral_mixer(image_mask, photon_mask, number_of_spectra, in_photon, s_factor_value):
    # TODO:
    # implement a threshold mask for each "original photon mask" and associate different background spectra in each part

    # agarose
    background_spectra = np.round(np.array([
        85.724904860, 134.77387063, 230.22967863, 10.211142220,
        164.34450976, 18.720671180, 80.654070700, 48.678445020,
        42.018761160, 20.546031480, 142.97878536, 17.177274700,
        7.1665455500, 10.176963810, 14.716390770, 39.074067910,
        19.990600590, 70.199832120, 178.09212828, 141.81554826,
        47.468315760, 88.611641250, 59.121639410, 135.13820124,
        77.332277620, 22.421808080, 116.27722422, 30.401546000,
        50.835330800, 31.002514360, 76.022144000, 23.136681870]))

    background_spectra = np.round(np.array([
        1.489938670, 2.520715180, 4.117703770, 6.494750020,
        9.891128970, 14.54472251, 20.65100664, 28.31087405,
        37.47500016, 47.89678546, 59.10816646, 70.43119185,
        81.03246394, 90.01803543, 96.55538770, 100.0000000,
        100.0000000, 96.55538770, 90.01803543, 81.03246394,
        70.43119185, 59.10816646, 47.89678546, 37.47500016,
        28.31087405, 20.65100664, 14.54472251, 9.891128970,
        6.494750020, 4.117703770, 2.520715180, 1.489938670]))

    num_extra_back_spectra = number_of_spectra - 1  # number of background spectra other than agarose
    back_conversion_factor = np.random.randint(40, 70, size=image_mask.shape)
    ph_low = 0  # minimum number of background spectra photons to simulate
    # maximum number of background spectra photons to simulate
    ph_high = in_photon * s_factor_value.mean() / back_conversion_factor.mean() * (1 /10 )

    # scaling background as photon/sqrt(photon) of signal
    #ph_high = in_photon * s_factor_value.mean() / back_conversion_factor.mean() * (image_mask.max() / np.sqrt(image_mask.max())) * 100
    # backspectra_list = ['gaussian', 'elastin', 'bd_unknown', 'Retinol']
    # backspectra_list = ['gaussian1', 'elastin', 'bd_unknown', 'gaussian2']
    backspectra_list = ['gaussian1', 'NADH_bound']#,'gaussian2']

    if number_of_spectra == 5:
        # TODO: Need to implement the photon mask for this option.  Currently not completed.
        # splits background in 4 parts, each with separate spectra and combinations
        # create 4 sections each with different indices:
        background_matrix = np.zeros(image_mask.shape, dtype="uint16")
        bm_y_half = np.int(background_matrix.shape[3] / 2)
        bm_x_half = np.int(background_matrix.shape[4] / 2)
        background_photon_mask_shape = (1,
                                        image_mask.shape[1], image_mask.shape[2],
                                        bm_y_half, bm_x_half)
        back_arr_process_shape = background_photon_mask_shape[1:] + (image_mask.shape[-1],)
        back_arr_save_shape = (image_mask.shape[0],) + back_arr_process_shape
        final_back_arr_save_shape = (num_extra_back_spectra,) + back_arr_save_shape

        # Set up multiprocessing shared array for full photon array
        back_size = np.prod(background_photon_mask_shape).item()
        uint16_ctype = np.ctypeslib.as_ctypes_type(np.uint16)
        back_full_arr_share = mp.RawArray(uint16_ctype, back_size)
        background_photon_mask = np.ctypeslib.as_array(back_full_arr_share).reshape(background_photon_mask_shape)
        final_stoch_back_arr = np.zeros(final_back_arr_save_shape, dtype="uint16")

        for n in range(num_extra_back_spectra):
            background_photon_mask[...] = np.random.randint(ph_low, ph_high + 1, size=background_photon_mask_shape)
            final_stoch_back_arr[n, ...], _ = \
                simulate_shape_spectra(back_full_arr_share,
                                       background_photon_mask_shape, back_arr_process_shape, back_arr_save_shape,
                                       [backspectra_list[n]], back_conversion_factor, image_mask.shape[0], 0)

        final_stoch_back_arr *= back_conversion_factor

        # set up slices for the four quadrants
        y_slices = [slice(0, bm_y_half), slice(bm_y_half, None)]
        x_slices = [slice(0, bm_x_half), slice(bm_x_half, None)]
        yx_slices = itertools.product(y_slices, x_slices)

        section_indeXes = np.zeros((4, num_extra_back_spectra), dtype="bool")
        for section_c, yx_slice in enumerate(yx_slices):
            section_indeXes[section_c, :] = np.random.randint(0, 2, size=num_extra_back_spectra).astype("bool")
            temp_mat = final_stoch_back_arr[section_indeXes[section_c, :], ...].sum(axis=0)
            background_matrix[:, :, :, yx_slice[0], yx_slice[1], :] = temp_mat

        aa = 0
    elif number_of_spectra > 0:
        # randomly pick spectra if less than 3
        #photon_mask[photon_mask == False] = True
        background_photon_mask_shape = (number_of_spectra,) + image_mask.shape[1:-1]
        back_arr_process_shape = background_photon_mask_shape[1:] + (image_mask.shape[-1],)
        back_arr_save_shape = (image_mask.shape[0],) + back_arr_process_shape

        # Set up multiprocessing shared array for full photon array
        back_size = np.prod(background_photon_mask_shape).item()
        uint16_ctype = np.ctypeslib.as_ctypes_type(np.uint16)
        back_full_arr_share = mp.RawArray(uint16_ctype, back_size)
        background_photon_mask = np.ctypeslib.as_array(back_full_arr_share).reshape(background_photon_mask_shape)
        background_photon_mask[np.tile(photon_mask, (number_of_spectra,) + (1,) * len(photon_mask.shape))] = \
            np.random.randint(ph_low, ph_high + 1, size=(number_of_spectra * np.count_nonzero(photon_mask), 1)).ravel()

        backspec_selected_list = backspectra_list[:number_of_spectra]
        background_matrix, _ = \
            simulate_shape_spectra(back_full_arr_share,
                                   background_photon_mask_shape, back_arr_process_shape, back_arr_save_shape,
                                   backspec_selected_list, back_conversion_factor.mean(), image_mask.shape[0], 0)

        background_matrix = background_matrix.astype("float")
        # back_conversion_factor = np.random.randint(80, 120, size=background_matrix.shape)
        background_matrix *= back_conversion_factor / s_factor_value.mean() / 100
    elif number_of_spectra == 0:
        background_matrix = np.zeros(image_mask.shape, dtype="uint16")
    else:
        raise (AssertionError('number_of_spectra cannot be greater 5'))

    return background_matrix


def image_PSF_poisson_readout_noise(true_spectra_save_arr, stochastic_arr, photon_full_arr, in_photon,
                                    s_factor_value, poisson_iteration, psf_iteration, back_spectra_number,
                                    sigtonoise_ratio):
    # This function:
    # 1. applies PSF
    # 2. applies poisson noise
    # 3. applies readout noise
    # 4. does average if needed
    # aa = 0
    # stochastic_arr_smooth = np.zeros(stochastic_arr.shape, dtype="float")
    Gauss2DKern_mat = gkern(3,.5)#(5, .5)
    # PSF_rounds = 3

    offset_detector = np.round(np.array([107.928285633891, 29.0939862818254, -12.9728756628311, 71.7258179561244,
                                         19.0854168148337, 77.8247534618967, 56.9442672706439, 68.9751985198219,
                                         59.9965470225510, 71.1252720767640, 26.2773677958737, 52.4748499741760,
                                         40.1233939877471, 37.7559220595540, 51.8744185950698, 34.1898085389329,
                                         76.0935930027346, 39.5083637387384, -74.2746874872085, -18.3023645830734,
                                         55.3971250552799, -3.10433480679051, 41.0388518649854, -1.37157552268344,
                                         56.2989492664762, 23.0682222522905, 40.0994914532690, 11.9054832062331,
                                         54.3889724609238, 7.94235787118964, 73.8963143471035, 35.7378697457512]))
    # This is obtained from center of gaussian shape of background + offset detector estimation
    new_center_background = np.round(np.array([95.6923413284774, 95.3869418955304, 98.4584767548588, 50.8303170228442,
                                               99.1411867530302, 62.3261684026210, 79.7525113753129, 74.0738944483111,
                                               65.4877816192333, 52.6249730219618, 71.7672928972851, 45.6145301252172,
                                               34.4975948719338, 32.1770904268678, 39.2280720314115, 53.4330338784193,
                                               57.7026295047071, 52.9834214399925, 66.4326005590467, 57.4265092671128,
                                               57.7522472702590, 53.9904459696171, 57.4217021756815, 56.5496218722914,
                                               54.6863175815921, 54.9711256753819, 54.9233394372773, 54.4165212188855,
                                               59.7988663345440, 60.2667060148365, 61.4620816476679, 54.5401489616441]))

    # sigma_readout_noise = 175#s_factor_value/3

    sigma_readout_noise = np.round(np.array([31.8974471094925, 31.7956472985101, 32.8194922516196, 16.9434390076147,
                                             33.0470622510101, 20.7753894675403, 26.5841704584376, 24.6912981494370,
                                             21.8292605397444, 17.5416576739873, 23.9224309657617, 15.2048433750724,
                                             11.4991982906446, 10.7256968089559, 13.0760240104705, 17.8110112928064,
                                             19.2342098349024, 17.6611404799975, 22.1442001863489, 19.1421697557043,
                                             19.2507490900863, 17.9968153232057, 19.1405673918938, 18.8498739574305,
                                             18.2287725271974, 18.3237085584606, 18.3077798124258, 18.1388404062952,
                                             19.9329554448480, 20.0889020049455, 20.4873605492226,
                                             18.1800496538814]))
    # agarose
    background_spectra = np.round(np.array([85.72490486, 134.77387063, 230.22967863, 10.21114222,
                                            164.34450976, 18.72067118, 80.6540707, 48.67844502,
                                            42.01876116, 20.54603148, 142.97878536, 17.1772747,
                                            7.16654555, 10.17696381, 14.71639077, 39.07406791,
                                            19.99060059, 70.19983212, 178.09212828, 141.81554826,
                                            47.46831576, 88.61164125, 59.12163941, 135.13820124,
                                            77.33227762, 22.42180808, 116.27722422, 30.401546,
                                            50.8353308, 31.00251436, 76.022144, 23.13668187]))

    # GAUSSIAN np.round(np.array([1.48993867, 2.52071518, 4.11770377, 6.49475002, 9.89112897, 14.54472251, 20.65100664, 28.31087405,
    # 37.47500016, 47.89678546, 59.10816646, 70.43119185,81.03246394, 90.01803543, 96.5553877, 100.,
    # 100., 96.5553877, 90.01803543, 81.03246394,
    # 70.43119185, 59.10816646, 47.89678546, 37.47500016,
    # 28.31087405, 20.65100664, 14.54472251, 9.89112897,
    # 6.49475002, 4.11770377, 2.52071518, 1.48993867]))

    # for img_ind in range(stochastic_arr.shape[0]):
    #     for chan_ind in range(stochastic_arr.shape[5]):
    #         # 1. apply PSF
    #         for i in range(PSF_rounds):
    #             for t in range(stochastic_arr.shape[1]):
    #                 for z in range(stochastic_arr.shape[2]):
    #                     # stochastic_arr_smooth[img_ind,t,z,:,:,chan_ind] = ndimage.convolve(stochastic_arr[img_ind,t,z,:,:,chan_ind],
    #                     #                                                                    Gauss2DKern_mat, mode='reflect')
    #                     aa = 0
    #                     stochastic_arr_smooth[img_ind, t, z, :, :, chan_ind] = stochastic_arr[img_ind,t,z,:,:,chan_ind]
    #
    # stochastic_arr_smooth[stochastic_arr_smooth<0]=0
    # 2. applies poisson noise
    # rng = RandomState(4213)

    poisson_arr = np.copy(stochastic_arr.astype("double"))

    photon_mask = photon_full_arr.sum(axis=0).astype("bool")

    background_spectra = background_spectral_mixer(stochastic_arr, photon_mask, back_spectra_number,
                                                   in_photon, s_factor_value)
    poisson_arr += background_spectra.astype("double")

    # #Gain noise
    # for gn in range(poisson_arr.shape[5]):
    #     gauss_noise = np.random.normal(0, .3, poisson_arr.shape[:-1])
    #     gauss_noise += np.abs(gauss_noise.min())
    #     poisson_arr[:,:,:,:,:,gn] +=  gauss_noise

    #gain_noise = np.random.normal(0, 6, poisson_arr.shape[:-1])

    #poisson_arr += gain_noise


    scaling_for_convolution = 1000
    gaussNDKern = Gauss2DKern_mat[np.newaxis, np.newaxis, :, :, np.newaxis]
    for img_ind in range(poisson_arr.shape[0]):
        for i in range(psf_iteration):
            poisson_arr[img_ind, :, :, :, :, :] = ndimage.convolve(
                120 + scaling_for_convolution * poisson_arr[img_ind, :, :, :, :, :],
                gaussNDKern, mode='reflect')
            poisson_arr[img_ind, :, :, :, :, :] -= 120
            poisson_arr[img_ind, :, :, :, :, :] /= scaling_for_convolution

    for tt in range(poisson_iteration):
        poisson_arr = np.random.poisson(10 * poisson_arr)

        poisson_arr = np.divide(poisson_arr, 10)

    # now photons are detected. They get converted to DL
    stochastic_arr_smooth = s_factor_value * poisson_arr.astype("float")

    # Gain noise
    #using photon conversion (variance/intensity) as a way to dynamically scale the standard deviation on a pixel basis
    zero_arr = np.zeros(stochastic_arr_smooth.shape[:-1])
    for gn in range(poisson_arr.shape[5]):
        gauss_noise = np.random.normal(zero_arr, np.sqrt( s_factor_value[gn]*stochastic_arr_smooth[:, :, :, :, :, gn]), stochastic_arr_smooth.shape[:-1])
        #gauss_noise += np.abs(gauss_noise.min())
        stochastic_arr_smooth[:, :, :, :, :, gn] += gauss_noise

    # stochastic_arr_smooth[stochastic_arr_smooth==0] = 200  # adding background from biological noise

    # optical filter
    fil = create_optical_filter_array()
    axis_expand = [None for el in stochastic_arr_smooth.shape]
    axis_expand[-1] = slice(None)
    axis_expand = tuple(axis_expand)
    final_spectrum_filter_arr = stochastic_arr_smooth * fil[axis_expand]

    # 3. applies readout noise
    # readout noise matrix is in DL
    # create array of readout noise

    background_arr = readout_noise_image(sigma_readout_noise, new_center_background, stochastic_arr.shape)

    final_spectrum_arr = stochastic_arr_smooth + background_arr
    final_spectrum_filter_arr = final_spectrum_filter_arr + background_arr

    stochastic_arr_smooth = stochastic_arr_smooth.astype("uint16")
    background_arr[background_arr < 0] = 0
    background_arr = background_arr.astype("uint16")

    # salt and pepper noise
    final_spectrum_arr = addsalt_pepper(final_spectrum_arr, sigtonoise_ratio)
    final_spectrum_filter_arr = addsalt_pepper(final_spectrum_filter_arr, sigtonoise_ratio)

    # subtract offset
    final_spectrum_arr -= offset_detector[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    final_spectrum_filter_arr -= offset_detector[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # check negatives
    final_spectrum_arr[final_spectrum_arr < 0] = 0
    final_spectrum_arr[final_spectrum_arr > 65535] = 65535
    final_spectrum_arr = final_spectrum_arr.astype("uint16")

    final_spectrum_filter_arr[final_spectrum_filter_arr < 0] = 0
    final_spectrum_filter_arr[final_spectrum_filter_arr > 65535] = 65535
    final_spectrum_filter_arr = final_spectrum_filter_arr.astype("uint16")

    true_scale = final_spectrum_arr.max(axis=-1) / true_spectra_save_arr.max(axis=-1)
    final_true_spectra_arr = true_spectra_save_arr * true_scale[:, :, :, :, :, np.newaxis]
    final_true_spectra_filter_arr = final_true_spectra_arr * fil

    final_true_spectra_arr = final_true_spectra_arr.astype("uint16")
    final_true_spectra_filter_arr = final_true_spectra_filter_arr.astype("uint16")

    final_spectrum_arr = final_spectrum_arr.mean(axis=0).astype("uint16")
    final_spectrum_filter_arr = final_spectrum_filter_arr.mean(axis=0).astype("uint16")
    final_true_spectra_arr = final_true_spectra_arr.mean(axis=0).astype("uint16")
    final_true_spectra_filter_arr = final_true_spectra_filter_arr.mean(axis=0).astype("uint16")
    print("finish")
    return final_spectrum_arr, final_true_spectra_arr, \
           final_spectrum_filter_arr, final_true_spectra_filter_arr, \
           background_arr, stochastic_arr_smooth, poisson_arr


def gkern(kernlen=10, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def readout_noise(sigma, mu):
    readout_noise_vect = np.random.normal(mu, sigma)
    if readout_noise_vect > 60000:
        print('outlier here a11')
    aa = np.round(readout_noise_vect)
    return aa


def readout_noise_image(sigma, mu, arr_shape):
    # readout_noise_vect = np.float(readout_noise_vect)
    readout_noise_vect = np.zeros(arr_shape, "float")
    sp_chan = arr_shape[-1]
    for i in range(sp_chan):
        readout_noise_vect[:, :, :, :, :, i] = np.random.normal(mu[i], sigma[i], arr_shape[:-1])
        if (readout_noise_vect > 60000).any():
            print('outlier here a11')
    aa = np.round(readout_noise_vect)
    return aa


if __name__ == "__main__":
    # TODO: Set Parameters
    in_dir = "C:/data/H2U-paper/H2U_shape_simulation_input_images/from_02-single_positive-tif_average_small"
    
    for j in [1]:  # [1,2,3,4,5]:
        for i in [20]: #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:#,25,30,35,40,45,50]:
            ftype = 'tif'
            out_fileprefix = '210217-'
            ph_max = i
            ph_avg = i
            back_spectra = 2
            ph_thresh = 1
            int_thresh_list = [82, 850, 1000, 500, 600]
            SNR = [.92]  # [0.9,0.8,0.6,0.4]
            
            #s_factor = (s827*sensitivity)/(QE*10)
            # Note: sensitivity is related to QE[%]=124*sensitivity/wavelength
            
            # s_factor = np.asarray([414.98035846, 408.14610265, 346.97051238, 478.61627781,
            #                        528.02459889, 446.559529, 463.40427699, 433.74800101,
            #                        460.78944279, 403.42818595, 446.04861064, 449.41756563,
            #                        470.62749018, 435.03099301, 452.76047381, 402.50295131,
            #                        410.19721778, 395.00371394, 404.31703527, 379.25287595,
            #                        436.02617246, 384.40369025, 443.89548083, 430.85655075,
            #                        451.7569713, 389.69904436, 452.83704071, 404.34056341,
            #                        488.88652996, 271.88868298, 378.85732929, 316.23585959])
            
            #s_factor = s827*QE/100
            
            # s_factor = np.asarray([438.66948458, 451.70991461, 385.41402665, 543.02429819,
            #        606.58133031, 515.3236138, 534.92697819, 497.84399574,
            #        524.23198925, 453.75504282, 495.20086388, 490.08422462,
            #        505.50312261, 459.10374564, 470.03972008, 409.98142774,
            #        408.99632813, 385.951905, 387.06487234, 355.4496321,
            #        403.14641205, 350.72152383, 393.58578602, 369.52618618,
            #        366.79708772, 304.23697625, 349.9648654, 301.16651467,
            #        329.71809956, 174.41941889, 226.83953393, 176.34496703])
            #
            #s-factor for 827gain
            
            #s827
            s_factor = np.asarray([1254.77541357, 1207.23177862, 1004.88613092, 1357.1875189,
                   1467.08588573, 1215.95944738, 1237.51209501, 1136.65607831,
                   1184.97285093, 1018.66703219, 1106.34688089, 1094.25553089,
                   1125.99261062, 1023.32325615, 1046.99897554, 915.34143277,
                   917.73174197, 869.59399995, 876.02949561, 809.12732096,
                   915.82556121, 795.43119802, 904.96134007, 865.76586426,
                   894.71433242, 760.59244062, 871.51326177, 767.89014449,
                   915.62926842, 502.31667451, 690.32116228, 568.57961318])
            s_factor_name = '32sep'
            number_avg_frames = 1
            poiss_noise_add = [5]
            psf = 2
            # Number of spectra types needs to match the number of files in in_dir
            # FP_list = ['NADH_bound', 'Retinol', 'NADH_free', 'Retinoic_acid','FAD','elastin','tdtomato','mko2','mruby','citrine']
            # FP_list = ['citrine']
            FP_list = ['mko2', 'citrine', 'mruby', 'tdtomato']
            
            cal_path = ""
            
            for pn in poiss_noise_add:
                for snr in SNR:
                    out_filename = out_fileprefix + '_po_' + str(pn) + '_ph_' + str(ph_thresh) + '_max' + str(ph_avg) + \
                                   '_sf_' + str(s_factor_name) + '_avg_' + str(number_avg_frames) + \
                                   '_PSF_' + str(psf) + '_bs_' + str(back_spectra) + \
                                   '_SNR_' + str(snr)
                    generate_photon_mask_simulation(out_filename, in_dir, cal_path, ftype, FP_list, ph_avg,
                                                    ph_thresh, int_thresh_list, s_factor, number_avg_frames,
                                                    pn, psf, back_spectra, snr)
