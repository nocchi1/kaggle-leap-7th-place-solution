# weightが0のターゲットカラム
ZERO_WEIGHT_TARGET_COLS = [
    "ptend_q0001_0",
    "ptend_q0001_1",
    "ptend_q0001_2",
    "ptend_q0001_3",
    "ptend_q0001_4",
    "ptend_q0001_5",
    "ptend_q0001_6",
    "ptend_q0001_7",
    "ptend_q0001_8",
    "ptend_q0001_9",
    "ptend_q0001_10",
    "ptend_q0001_11",
    "ptend_q0002_0",
    "ptend_q0002_1",
    "ptend_q0002_2",
    "ptend_q0002_3",
    "ptend_q0002_4",
    "ptend_q0002_5",
    "ptend_q0002_6",
    "ptend_q0002_7",
    "ptend_q0002_8",
    "ptend_q0002_9",
    "ptend_q0002_10",
    "ptend_q0002_11",
    "ptend_q0003_0",
    "ptend_q0003_1",
    "ptend_q0003_2",
    "ptend_q0003_3",
    "ptend_q0003_4",
    "ptend_q0003_5",
    "ptend_q0003_6",
    "ptend_q0003_7",
    "ptend_q0003_8",
    "ptend_q0003_9",
    "ptend_q0003_10",
    "ptend_q0003_11",
    "ptend_u_0",
    "ptend_u_1",
    "ptend_u_2",
    "ptend_u_3",
    "ptend_u_4",
    "ptend_u_5",
    "ptend_u_6",
    "ptend_u_7",
    "ptend_u_8",
    "ptend_u_9",
    "ptend_u_10",
    "ptend_u_11",
    "ptend_v_0",
    "ptend_v_1",
    "ptend_v_2",
    "ptend_v_3",
    "ptend_v_4",
    "ptend_v_5",
    "ptend_v_6",
    "ptend_v_7",
    "ptend_v_8",
    "ptend_v_9",
    "ptend_v_10",
    "ptend_v_11",
]
# 後処理を適用するターゲットカラム
PP_TARGET_COLS = [
    "ptend_q0002_12",
    "ptend_q0002_13",
    "ptend_q0002_14",
    "ptend_q0002_15",
    "ptend_q0002_16",
    "ptend_q0002_17",
    "ptend_q0002_18",
    "ptend_q0002_19",
    "ptend_q0002_20",
    "ptend_q0002_21",
    "ptend_q0002_22",
    "ptend_q0002_23",
    "ptend_q0002_24",
    "ptend_q0002_25",
    "ptend_q0002_26",
    # 'ptend_q0002_27',
]
# 垂直方向に次元をもつ入力特徴量
VERTICAL_INPUT_COLS = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
    # 追加した特徴量
    "state_ice_rate",
    "state_ice_rate_diff",
    "state_dp",
    "state_rh",
    "state_vp",
]
# サンプルごとに単一の値をもつ入力特徴量
SCALER_INPUT_COLS = [
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
    "pbuf_TAUX",
    "pbuf_TAUY",
    "pbuf_COSZRS",
    "cam_in_ALDIF",
    "cam_in_ALDIR",
    "cam_in_ASDIF",
    "cam_in_ASDIR",
    "cam_in_LWUP",
    "cam_in_ICEFRAC",
    "cam_in_LANDFRAC",
    "cam_in_OCNFRAC",
    "cam_in_SNOWHLAND",
]
# 垂直方向に次元をもつターゲット
VERTICAL_TARGET_COLS = [
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
]
# サンプルごとに単一の値をもつターゲット
SCALER_TARGET_COLS = ["cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC", "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD"]

# old_sub_factorをかけた後のmin,max -> clippingに使用
TARGET_MIN_MAX = {
    "ptend_t_0": [-62.895582026739646, 245.9913771015123],
    "ptend_t_1": [-85.19733458669685, 485.2134138330683],
    "ptend_t_2": [-126.66634799367836, 67.96046403350691],
    "ptend_t_3": [-63.68305517812135, 2.831209156478813],
    "ptend_t_4": [-3.7250615866204653, 2.10641390616558],
    "ptend_t_5": [-2.5984165628028726, 2.273886463169375],
    "ptend_t_6": [-2.0388026382606115, 2.654018387635486],
    "ptend_t_7": [-2.5411872014572965, 2.976271184131482],
    "ptend_t_8": [-2.526668527322188, 3.159980987118253],
    "ptend_t_9": [-2.4513562614448934, 3.3719923158473653],
    "ptend_t_10": [-2.545370142831551, 3.3882850980675046],
    "ptend_t_11": [-2.584272177781589, 3.3149029304104043],
    "ptend_t_12": [-2.602380790720544, 3.3947578481801766],
    "ptend_t_13": [-2.941554488500255, 3.241369613721731],
    "ptend_t_14": [-11.149694372476999, 6.4068403795561215],
    "ptend_t_15": [-60.62043304108669, 28.326355196318737],
    "ptend_t_16": [-289.51846356119455, 112.44741478518792],
    "ptend_t_17": [-214.07967560927975, 322.56339359024645],
    "ptend_t_18": [-213.8828958881219, 73.44115437093622],
    "ptend_t_19": [-152.01505522146422, 90.33332042208895],
    "ptend_t_20": [-105.21809120747639, 126.61560993380817],
    "ptend_t_21": [-79.60538403492549, 85.64140633890399],
    "ptend_t_22": [-61.91548536861084, 116.01953630732135],
    "ptend_t_23": [-56.61089440941644, 91.76161820839307],
    "ptend_t_24": [-36.65476768446294, 73.58117207956529],
    "ptend_t_25": [-34.77627434172876, 71.45026989542421],
    "ptend_t_26": [-26.41475739813556, 45.992454893094695],
    "ptend_t_27": [-31.008555506459977, 39.74139585813688],
    "ptend_t_28": [-21.78869208726987, 34.94322242965742],
    "ptend_t_29": [-20.32712112360059, 35.15368751894109],
    "ptend_t_30": [-13.413499376861804, 33.11287279602675],
    "ptend_t_31": [-18.306476795689154, 28.1562669036818],
    "ptend_t_32": [-19.504523139289795, 29.652583649795528],
    "ptend_t_33": [-21.31650800672111, 27.864006465018875],
    "ptend_t_34": [-19.8732089224267, 24.21690746439272],
    "ptend_t_35": [-24.41553233010459, 22.414743293163827],
    "ptend_t_36": [-25.15031425474549, 20.852196923863303],
    "ptend_t_37": [-20.980784998056325, 21.553742138061107],
    "ptend_t_38": [-16.788397636652096, 23.920526036526887],
    "ptend_t_39": [-16.54347363657158, 22.517418291569207],
    "ptend_t_40": [-13.980159270231189, 19.798294855743514],
    "ptend_t_41": [-15.32818295570466, 19.688572756213727],
    "ptend_t_42": [-12.283203771724894, 19.918201897887343],
    "ptend_t_43": [-11.303526004451566, 20.71533761663488],
    "ptend_t_44": [-9.88390611218265, 20.846160112232802],
    "ptend_t_45": [-11.268348704518754, 19.905809963995726],
    "ptend_t_46": [-10.65177854260462, 22.351331746164934],
    "ptend_t_47": [-11.341812815720905, 23.998290243171503],
    "ptend_t_48": [-11.833104950546238, 22.876150221052416],
    "ptend_t_49": [-15.680330226188437, 22.387904282614194],
    "ptend_t_50": [-15.540193408897444, 23.781678834625797],
    "ptend_t_51": [-21.481172106641186, 24.029521583107872],
    "ptend_t_52": [-21.58317313544652, 26.994590186489294],
    "ptend_t_53": [-23.85037566805397, 26.282351487743938],
    "ptend_t_54": [-20.861445379755963, 34.965566974726904],
    "ptend_t_55": [-20.881405685117187, 30.871708555426398],
    "ptend_t_56": [-23.01427491919194, 23.13489805060605],
    "ptend_t_57": [-28.159035454164837, 21.92397606256335],
    "ptend_t_58": [-32.231630826690726, 19.762423175263432],
    "ptend_t_59": [-24.383488182528488, 13.610232871497633],
    "ptend_q0001_0": [-1.1190786040668127e-11, 1.8081592561702364e-11],
    "ptend_q0001_1": [-1.0895678024328868e-11, 1.480026387473359e-11],
    "ptend_q0001_2": [-1.0685745747878693e-11, 1.4009461564244229e-11],
    "ptend_q0001_3": [-8.02824111672893e-12, 2.7747944169032014e-12],
    "ptend_q0001_4": [0.0, 0.0],
    "ptend_q0001_5": [0.0, 0.0],
    "ptend_q0001_6": [0.0, 0.0],
    "ptend_q0001_7": [0.0, 0.0],
    "ptend_q0001_8": [0.0, 0.0],
    "ptend_q0001_9": [0.0, 0.0],
    "ptend_q0001_10": [0.0, 0.0],
    "ptend_q0001_11": [0.0, 0.0],
    "ptend_q0001_12": [-89.12280378649038, 45.713430494499235],
    "ptend_q0001_13": [-95.05895245471501, 112.39978656810428],
    "ptend_q0001_14": [-96.56629170051266, 1621.390104573609],
    "ptend_q0001_15": [-322.37162391918, 565.4019201470468],
    "ptend_q0001_16": [-349.39643141691795, 522.6546172579565],
    "ptend_q0001_17": [-563.5537212962934, 816.6412654147047],
    "ptend_q0001_18": [-306.16894549113783, 936.045581703674],
    "ptend_q0001_19": [-192.04484437374464, 367.1627425057771],
    "ptend_q0001_20": [-92.53899180137167, 154.4966617002766],
    "ptend_q0001_21": [-90.85010173045147, 96.2027943795342],
    "ptend_q0001_22": [-67.66543622280179, 66.39108485106763],
    "ptend_q0001_23": [-54.82293702160418, 58.79436251234311],
    "ptend_q0001_24": [-37.27114768667174, 45.970048151088754],
    "ptend_q0001_25": [-38.992229541529035, 36.46415299044067],
    "ptend_q0001_26": [-39.1644676034942, 33.25417053143742],
    "ptend_q0001_27": [-35.7927767906358, 24.980815191068857],
    "ptend_q0001_28": [-30.695256549961606, 23.60178207367653],
    "ptend_q0001_29": [-30.045286317009584, 21.688328440866677],
    "ptend_q0001_30": [-25.98333403642693, 19.78728643280756],
    "ptend_q0001_31": [-25.436343929907995, 21.114857036478746],
    "ptend_q0001_32": [-27.907525478741608, 26.5556610014655],
    "ptend_q0001_33": [-26.48980226601897, 28.620967351366588],
    "ptend_q0001_34": [-26.04854625303789, 22.49661918302992],
    "ptend_q0001_35": [-28.561103384801307, 24.351164285552002],
    "ptend_q0001_36": [-30.448989004437568, 25.19974169004323],
    "ptend_q0001_37": [-27.937039890187812, 32.31019262291435],
    "ptend_q0001_38": [-23.949665644301845, 27.91172399588334],
    "ptend_q0001_39": [-29.610518238387566, 30.305101440493814],
    "ptend_q0001_40": [-25.344883244691726, 25.657326945180728],
    "ptend_q0001_41": [-21.392185443063706, 21.83275584396731],
    "ptend_q0001_42": [-21.389337866762276, 22.123208431064],
    "ptend_q0001_43": [-18.644385537779613, 21.371657241328986],
    "ptend_q0001_44": [-18.676671255965026, 19.44975953645763],
    "ptend_q0001_45": [-16.30001558552254, 17.420945443724296],
    "ptend_q0001_46": [-18.52898493076917, 16.06792099790536],
    "ptend_q0001_47": [-18.053237256832475, 15.512644466223717],
    "ptend_q0001_48": [-17.433764883337552, 14.279700737699],
    "ptend_q0001_49": [-16.019108788979167, 15.222358856405538],
    "ptend_q0001_50": [-14.878382643099881, 14.372289471899576],
    "ptend_q0001_51": [-15.646247178584552, 12.74094616697781],
    "ptend_q0001_52": [-15.24979541410117, 13.429652509007578],
    "ptend_q0001_53": [-15.384282608212091, 11.292564650852631],
    "ptend_q0001_54": [-15.193623985708964, 11.007335095893717],
    "ptend_q0001_55": [-17.61627699437948, 10.969354099682086],
    "ptend_q0001_56": [-21.282536551558675, 10.333361323436899],
    "ptend_q0001_57": [-23.390853089565844, 10.826288565207465],
    "ptend_q0001_58": [-21.819170956255583, 13.15840608794704],
    "ptend_q0001_59": [-25.893706977723628, 17.55532038340892],
    "ptend_q0002_0": [0.0, 0.0],
    "ptend_q0002_1": [0.0, 0.0],
    "ptend_q0002_2": [0.0, 0.0],
    "ptend_q0002_3": [0.0, 0.0],
    "ptend_q0002_4": [0.0, 0.0],
    "ptend_q0002_5": [0.0, 0.0],
    "ptend_q0002_6": [0.0, 0.0],
    "ptend_q0002_7": [0.0, 0.0],
    "ptend_q0002_8": [0.0, 0.0],
    "ptend_q0002_9": [0.0, 0.0],
    "ptend_q0002_10": [0.0, 0.0],
    "ptend_q0002_11": [0.0, 0.0],
    "ptend_q0002_12": [-2.692638476909033e-38, -2.1473569287542685e-53],
    "ptend_q0002_13": [-1.8047277675416653e-43, 0.0],
    "ptend_q0002_14": [-3.4568419634524655e-48, 0.0],
    "ptend_q0002_15": [-6.381237267561655e-33, 0.0],
    "ptend_q0002_16": [-1.9592021055394953e-29, 0.0],
    "ptend_q0002_17": [-1.3947599115989293e-24, 0.0],
    "ptend_q0002_18": [-8.14260051935312e-22, 0.0],
    "ptend_q0002_19": [-1.9362630451901274e-19, 0.0],
    "ptend_q0002_20": [-1.6097461535421175e-16, 0.0],
    "ptend_q0002_21": [-1.1756553055974902e-13, 0.0],
    "ptend_q0002_22": [-9.266323275947731e-11, 0.0],
    "ptend_q0002_23": [-6.226475179289799e-08, 0.0],
    "ptend_q0002_24": [-0.00017506402040736702, 0.0],
    "ptend_q0002_25": [-6.1044498114859955, 0.0],
    "ptend_q0002_26": [-3053.2581105598542, 0.0],
    "ptend_q0002_27": [-1020.7392037921659, 2523.089684990934],
    "ptend_q0002_28": [-203.96595655105665, 184.53086032823535],
    "ptend_q0002_29": [-119.35668372825386, 69.4924333425878],
    "ptend_q0002_30": [-86.3857589847951, 61.14787214595843],
    "ptend_q0002_31": [-58.36718909480134, 58.55804562439231],
    "ptend_q0002_32": [-56.3035934819871, 45.23304211458134],
    "ptend_q0002_33": [-45.99828671496109, 38.78199830776173],
    "ptend_q0002_34": [-36.82652920099126, 38.55216487084694],
    "ptend_q0002_35": [-30.58930464532066, 35.23991820673057],
    "ptend_q0002_36": [-30.73506478317152, 36.752375711979816],
    "ptend_q0002_37": [-22.998223810135766, 36.2249007310614],
    "ptend_q0002_38": [-25.308332316979246, 33.94380832636659],
    "ptend_q0002_39": [-19.771956494125266, 30.513649764844473],
    "ptend_q0002_40": [-21.141683174403582, 31.768425208155616],
    "ptend_q0002_41": [-17.34801020096885, 28.879748616518654],
    "ptend_q0002_42": [-15.99129183227161, 28.178896683314978],
    "ptend_q0002_43": [-19.19078666869037, 25.626428819004538],
    "ptend_q0002_44": [-18.08532183517627, 25.66184475515433],
    "ptend_q0002_45": [-12.479532454358255, 20.660700608079534],
    "ptend_q0002_46": [-12.787798804666528, 19.775196917930707],
    "ptend_q0002_47": [-10.784452380372862, 23.738416227013357],
    "ptend_q0002_48": [-14.792382025736682, 19.004223075262832],
    "ptend_q0002_49": [-10.99927658580884, 20.363318315343417],
    "ptend_q0002_50": [-14.088684703792289, 30.47508411264497],
    "ptend_q0002_51": [-11.749279331039723, 27.144752544860996],
    "ptend_q0002_52": [-13.505169503148043, 32.35483608191819],
    "ptend_q0002_53": [-10.90931296174185, 31.149397115147757],
    "ptend_q0002_54": [-15.873690649973721, 35.1307721560634],
    "ptend_q0002_55": [-16.984043180275812, 38.93718840257822],
    "ptend_q0002_56": [-16.879141997493655, 42.39623072555317],
    "ptend_q0002_57": [-20.172938724022206, 51.084542534661104],
    "ptend_q0002_58": [-20.088161388436433, 46.051795354611286],
    "ptend_q0002_59": [-32.902372531508135, 57.07812028417396],
    "ptend_q0003_0": [0.0, 0.0],
    "ptend_q0003_1": [0.0, 0.0],
    "ptend_q0003_2": [0.0, 0.0],
    "ptend_q0003_3": [0.0, 0.0],
    "ptend_q0003_4": [0.0, 0.0],
    "ptend_q0003_5": [0.0, 0.0],
    "ptend_q0003_6": [0.0, 0.0],
    "ptend_q0003_7": [0.0, 0.0],
    "ptend_q0003_8": [0.0, 0.0],
    "ptend_q0003_9": [0.0, 0.0],
    "ptend_q0003_10": [0.0, 0.0],
    "ptend_q0003_11": [0.0, 0.0],
    "ptend_q0003_12": [-46.02251335691983, 84.88144854329252],
    "ptend_q0003_13": [-41.825074754822154, 87.59475110688157],
    "ptend_q0003_14": [-413.6201264485793, 829.2320010498253],
    "ptend_q0003_15": [-535.6887229734319, 1270.3626859746346],
    "ptend_q0003_16": [-304.5092957419094, 1115.1421211040877],
    "ptend_q0003_17": [-132.91775176305063, 686.360247624323],
    "ptend_q0003_18": [-136.7376290546697, 369.4685682319266],
    "ptend_q0003_19": [-85.74578379576265, 243.48120072891393],
    "ptend_q0003_20": [-72.06984761684221, 181.5681896106205],
    "ptend_q0003_21": [-67.05144248712507, 126.04813329607957],
    "ptend_q0003_22": [-57.487184768697745, 107.30988770849918],
    "ptend_q0003_23": [-41.98030552967116, 95.72291170188717],
    "ptend_q0003_24": [-39.29491330525031, 81.67867963779628],
    "ptend_q0003_25": [-35.53259719017107, 74.73368209277905],
    "ptend_q0003_26": [-35.92830453072824, 69.35125569801286],
    "ptend_q0003_27": [-32.60472132885498, 85.8650747158155],
    "ptend_q0003_28": [-30.66968325310654, 70.83540832269591],
    "ptend_q0003_29": [-28.664931165813993, 58.22506353241495],
    "ptend_q0003_30": [-25.924314413114967, 47.59179423287238],
    "ptend_q0003_31": [-23.006101443755576, 47.61890601966047],
    "ptend_q0003_32": [-22.862277393921286, 51.433666922003674],
    "ptend_q0003_33": [-23.83856323151652, 45.00387437012798],
    "ptend_q0003_34": [-26.566202771866802, 44.7024171307201],
    "ptend_q0003_35": [-30.425350128529747, 54.79463925933516],
    "ptend_q0003_36": [-24.296072949429256, 55.47626699310015],
    "ptend_q0003_37": [-25.303912161780758, 51.15016941382516],
    "ptend_q0003_38": [-22.565636996218, 43.11790717766895],
    "ptend_q0003_39": [-23.32121243073491, 47.791852138419465],
    "ptend_q0003_40": [-23.904132485626373, 43.01142996561283],
    "ptend_q0003_41": [-27.639424317054182, 42.67063553781691],
    "ptend_q0003_42": [-29.296133264986697, 37.48439006180556],
    "ptend_q0003_43": [-27.725268064887263, 44.88551557976122],
    "ptend_q0003_44": [-21.706150865639703, 46.05973810083105],
    "ptend_q0003_45": [-21.18003439480503, 46.88097082696383],
    "ptend_q0003_46": [-22.129801578809165, 34.69351406423478],
    "ptend_q0003_47": [-22.74196554797596, 38.510562129381015],
    "ptend_q0003_48": [-21.74882615266842, 42.31169504247042],
    "ptend_q0003_49": [-16.686641942251516, 42.01625962484335],
    "ptend_q0003_50": [-20.510362349833642, 39.28660611643692],
    "ptend_q0003_51": [-26.704120515263764, 39.00913435080203],
    "ptend_q0003_52": [-29.58581342157672, 45.027063668687745],
    "ptend_q0003_53": [-32.28377877532619, 40.42295539436537],
    "ptend_q0003_54": [-33.87658626534544, 33.54363254790539],
    "ptend_q0003_55": [-34.19369776274197, 34.93509912055992],
    "ptend_q0003_56": [-33.34041034492434, 28.571103887634404],
    "ptend_q0003_57": [-32.41123545965986, 24.413877822420034],
    "ptend_q0003_58": [-34.64892056107986, 22.56604733559916],
    "ptend_q0003_59": [-33.385938537637045, 24.035228403093846],
    "ptend_u_0": [0.0, 0.0],
    "ptend_u_1": [0.0, 0.0],
    "ptend_u_2": [0.0, 0.0],
    "ptend_u_3": [0.0, 0.0],
    "ptend_u_4": [0.0, 0.0],
    "ptend_u_5": [0.0, 0.0],
    "ptend_u_6": [0.0, 0.0],
    "ptend_u_7": [0.0, 0.0],
    "ptend_u_8": [0.0, 0.0],
    "ptend_u_9": [0.0, 0.0],
    "ptend_u_10": [0.0, 0.0],
    "ptend_u_11": [0.0, 0.0],
    "ptend_u_12": [-79.6917767371868, 77.33608541660243],
    "ptend_u_13": [-71.87845862955962, 72.19586744161872],
    "ptend_u_14": [-89.29269243474059, 56.0236155115297],
    "ptend_u_15": [-147.46604728838406, 44.37441779952108],
    "ptend_u_16": [-91.54572391020078, 210.34353080946343],
    "ptend_u_17": [-54.781538077059906, 298.4611702472299],
    "ptend_u_18": [-52.78624394540224, 473.72719996800737],
    "ptend_u_19": [-52.617903675187705, 310.2015798056361],
    "ptend_u_20": [-82.91330256834672, 160.98475798164154],
    "ptend_u_21": [-96.22737464287835, 90.28954849329556],
    "ptend_u_22": [-85.0928695026371, 88.31676321168676],
    "ptend_u_23": [-92.34070717531345, 92.43596798167846],
    "ptend_u_24": [-95.27678475658176, 114.99257145857611],
    "ptend_u_25": [-101.5182706053778, 107.69883429979052],
    "ptend_u_26": [-91.81599592005676, 66.9377450946793],
    "ptend_u_27": [-90.66179672466652, 63.09025211373046],
    "ptend_u_28": [-84.27371978079297, 87.57790177798353],
    "ptend_u_29": [-95.44455130766765, 84.61345321502662],
    "ptend_u_30": [-97.6288126416684, 89.51532713730036],
    "ptend_u_31": [-70.23993152577779, 94.16042469909452],
    "ptend_u_32": [-61.39160095420065, 84.6318927853533],
    "ptend_u_33": [-73.79749544499666, 71.98227499667662],
    "ptend_u_34": [-55.189480527181914, 92.96734255869384],
    "ptend_u_35": [-51.00651281110674, 52.5993430003381],
    "ptend_u_36": [-56.63540873336217, 61.938016302092684],
    "ptend_u_37": [-45.8241120400455, 55.91550934806791],
    "ptend_u_38": [-48.28598029035204, 48.37992155093106],
    "ptend_u_39": [-45.96961386731694, 61.901421963680455],
    "ptend_u_40": [-46.928984497206926, 55.80407926055752],
    "ptend_u_41": [-48.09145032831775, 45.034150286145966],
    "ptend_u_42": [-44.19460301436372, 37.53553368120117],
    "ptend_u_43": [-37.63453014674976, 61.0574359954419],
    "ptend_u_44": [-45.38164660310006, 38.918233893052935],
    "ptend_u_45": [-45.9229528396788, 36.55326728612385],
    "ptend_u_46": [-39.50864155703623, 27.748312947745625],
    "ptend_u_47": [-40.55063048480703, 30.972605496080806],
    "ptend_u_48": [-43.922788464699735, 26.870639399472513],
    "ptend_u_49": [-45.44690858238259, 23.866016110233407],
    "ptend_u_50": [-39.736366584078446, 25.42009988629471],
    "ptend_u_51": [-34.02857847377248, 21.653038467988466],
    "ptend_u_52": [-36.86345249152253, 26.11157598990839],
    "ptend_u_53": [-28.258373637294866, 25.930030756600583],
    "ptend_u_54": [-26.663845214483487, 55.87769173470087],
    "ptend_u_55": [-16.347715292846868, 38.71582934491053],
    "ptend_u_56": [-14.875114035880335, 36.743467971419626],
    "ptend_u_57": [-15.960725780788007, 28.537343612290055],
    "ptend_u_58": [-18.161174095261995, 25.389287572387047],
    "ptend_u_59": [-15.131567589916978, 21.324046401119023],
    "ptend_v_0": [0.0, 0.0],
    "ptend_v_1": [0.0, 0.0],
    "ptend_v_2": [0.0, 0.0],
    "ptend_v_3": [0.0, 0.0],
    "ptend_v_4": [0.0, 0.0],
    "ptend_v_5": [0.0, 0.0],
    "ptend_v_6": [0.0, 0.0],
    "ptend_v_7": [0.0, 0.0],
    "ptend_v_8": [0.0, 0.0],
    "ptend_v_9": [0.0, 0.0],
    "ptend_v_10": [0.0, 0.0],
    "ptend_v_11": [0.0, 0.0],
    "ptend_v_12": [-85.85078208508699, 71.63700905440317],
    "ptend_v_13": [-87.26775850425659, 68.35229793098478],
    "ptend_v_14": [-50.752309910443685, 52.36186998322725],
    "ptend_v_15": [-53.38614439985196, 117.2949998819104],
    "ptend_v_16": [-99.32525289760827, 74.78288568886357],
    "ptend_v_17": [-106.05647599158603, 93.16409912328704],
    "ptend_v_18": [-86.95913245693441, 95.95246067327928],
    "ptend_v_19": [-120.08763180307325, 178.4500492110286],
    "ptend_v_20": [-70.84136504285186, 132.20205026927306],
    "ptend_v_21": [-169.45237457559418, 115.49702067634409],
    "ptend_v_22": [-148.2321504762398, 73.79242368189293],
    "ptend_v_23": [-117.550424839033, 129.83321142986276],
    "ptend_v_24": [-106.77340039647258, 177.15619839296426],
    "ptend_v_25": [-115.67527706735032, 111.3772419236571],
    "ptend_v_26": [-100.19782644886132, 104.10402873472846],
    "ptend_v_27": [-120.63711784602597, 99.78471299811869],
    "ptend_v_28": [-125.76278693486476, 90.18832773920712],
    "ptend_v_29": [-77.52051548212972, 97.21042692752279],
    "ptend_v_30": [-96.96350745202056, 90.76030515341529],
    "ptend_v_31": [-81.50097120917133, 88.56213400685763],
    "ptend_v_32": [-63.42897147081308, 69.44212018455707],
    "ptend_v_33": [-64.09031636828377, 60.73731882572591],
    "ptend_v_34": [-91.24449329182403, 64.11486624511271],
    "ptend_v_35": [-83.75546360822635, 62.674207919687106],
    "ptend_v_36": [-90.0313078120674, 77.98324134576751],
    "ptend_v_37": [-72.6097807482047, 67.81759024531279],
    "ptend_v_38": [-60.86772285417931, 61.92118860399334],
    "ptend_v_39": [-59.17364788378461, 51.42769607053553],
    "ptend_v_40": [-55.50425442901407, 43.77778668067689],
    "ptend_v_41": [-50.70408866054272, 48.80369035190105],
    "ptend_v_42": [-54.748923259924126, 54.18960882112859],
    "ptend_v_43": [-41.703632295631365, 52.16651419103428],
    "ptend_v_44": [-44.097033957510924, 58.19335066333756],
    "ptend_v_45": [-39.83721864097757, 39.71707808318796],
    "ptend_v_46": [-28.736242897049575, 49.25761827839664],
    "ptend_v_47": [-27.97495281111826, 38.867008250322876],
    "ptend_v_48": [-26.61399328481802, 39.1197079032622],
    "ptend_v_49": [-28.022445881224318, 41.52864541723867],
    "ptend_v_50": [-27.708608370274263, 33.74144540845132],
    "ptend_v_51": [-34.57955837824143, 34.96846018000323],
    "ptend_v_52": [-38.41749954861223, 20.715256697848314],
    "ptend_v_53": [-40.31706845546787, 24.218038928226473],
    "ptend_v_54": [-39.00962182050916, 20.64331877756654],
    "ptend_v_55": [-22.825466820479217, 20.630961483300197],
    "ptend_v_56": [-19.97362593017722, 19.645103710719457],
    "ptend_v_57": [-17.629147803967637, 17.625199817138917],
    "ptend_v_58": [-22.6856061841494, 20.654040311696473],
    "ptend_v_59": [-22.50664009566739, 21.33009235861157],
    "cam_out_NETSW": [0.0, 4.486060091698915],
    "cam_out_FLWDS": [0.794318399052335, 7.331281165438898],
    "cam_out_PRECSC": [0.0, 39.66166772327446],
    "cam_out_PRECC": [0.0, 26.891735541099248],
    "cam_out_SOLS": [0.0, 4.706721719997375],
    "cam_out_SOLL": [0.0, 4.9413942198902605],
    "cam_out_SOLSD": [0.0, 9.15693365414901],
    "cam_out_SOLLD": [0.0, 9.116649336822254],
}
