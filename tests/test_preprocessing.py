from unittest import TestCase
from propythia.sequence import ReadSequence
import pandas as pd

class Test_preprocessing(TestCase):

    # ################# PHYSICO CHEMICAL DESCRIPTORS  ##################

    def setUp(self):
        test_data = {'sequence': {0: 'MSYKPIAPAPSSTPGSSTPGPGTPVPTGSVPSPSGSVPGAGAPFRPLFNDFGPPSMGYVQAMKPPGAQGSQSTYTDLLSVIEEMGKEIRPTYAGSKSAMERLKRGIIHARALVRECLAETERNART',
                                  1: 'MSDDLPIDIHSSKLLDWLVSRRHCNKDWQKSVVAIREKIKHAILDMPESPKIVELLQGAYINYFHCCQIIEILRDTEKDTKNFLGFYSSQRMKDWQEIEGMYKKDNVYLAEAAQILQRLAQYEIPALRKQISKMDQSVTDAIRKHSEYGKQAEDGRKQFEKEISRMQLKGVHLRKELLELAADLPAFYEKITAEIRKISAARDYFQAFRDYMSLGAAPKDAAPILPIIGLIGERGLDVTTYEWKYNQKPDKVEKPNFEMLLTAAEDSDEIDFGGGDEIDFGIAAEDDAVIDFSAVVDLVADDTGAVGEAIASGQDALHLLENSEAQKAVKHELIELLAFLSMRLDDETRETTADVLIRGAEKRPDGVAAVTEKRLKTWITEVEGILKELENPQKVHLFKIRGSPQYVEQVVEELEKKRDMEHRYKRLQTLMTENQETARQSVTKSNVELKTIVESTRVLQKQIEAEISKKYNGRRVNLMGGINQALGGN',
                                  2: 'MPFDPAASPLSPSQARVLATLMEKARTVPDSYPMSLNGLLTGCNQKTSRDPVMALSEAQVQEALAALERLALVFENSGYRSPRWEHNFQRGAGVPEQSAVLLGLLMLRGPQTAAELRTNAERWYRFADISSVEAFLDELQQRSADKGGPLAVPLPRSPGTREQRWAHLLCGPVDAGRSNAGVEPVPAGVETLQERIGTLESELASLRATVQWLCQELGITPAPASMPQPGLPAGNGSPGS',
                                  3: 'MIHFTKMHGLGNDFMVVDGVTQNVFFSPEQIRRLADRNFGIGFDQLLLVEPPYDPDLDFHYRIFNADGSEVEQCGNGARCFARFVRNKGLTQKNKIRVSTNSGKITLRIERDGNVTVNMGVPVIEPSQIPFKAKKSEKTYLLQTPMQTYLCGAISMGNPHCVIQVEDVQTVNVDEIGSSLTRHERFPKGVNVGFMQVINPGHIKLRVYERGAAETLACGTGACAAAAVGQLQDKLDKQVRVDLPGGSLIINWEGEGKPLWMTGPAEHVYDGQIQL',
                                  4: 'MGSSTTEPDVGTTSNIETTTTLQNKNVNEVDQNKKSEQSNPSFKEVVLKDIGLGEATDLENVSDDVFNNYLAIRLERERKEIELLKESNLEKLSVIIKNCIECNSFTDETMKRLINLVDNNYNNSVHFSPKSKRRKLESTSPPMSSSSVPNKETNNIQQSNSYQLRNNYDEEQENQKSQTQGSKSLLSRPNIYSFPPKQTQPASQQHVQLAAIVQRQSTLTTPLSSTYGSNSNNSMNTQLPLSDKSLRSNVQEKIVQQGSMSQRDIINGSNMSSQYSSQVYPPGYYQTRYGQQMVVVYPDSDSPQINQTSTIQHQQQLPHTYPPHYHQQQQLHQNQLVPQQHQQLQQQSISKHQLFGQKNPMSPQSHYLPNNESQNLGVINHRRSFSSGTYPVVNSRSKSPDRSMPLTVQKQMNFLIHTPKHPPPT',
                                  5: 'MSTNPIQPLLDVLYQGKSLNREQTAELFGALIRGEMSEAAMAGMLVALKMRGETIDEISGAADAMRAAAKPFPCPERNNNPLHNGIVDIVGTGGDGFNTINISTTAAFVAAAAGAKVAKHGNRSVSSKSGSSDLLAQFGIDLTMSPETASRCLDALNLCFLFAPHYHGGVKHAVPVRQALKTRTLFNVLGPLINPARPEFMLLGVYSPELVLPIAKVLKALGTKRAMVVHGSGLDEVALHGNTQVAELKDGDIVEYQLTPADLGVPLAQITDLEGGEPAQNALITEAILKGRGTEAHANAVAINAGCALYVCGIADSVKAGTLLALATIQSGKAFELLSQLAKVSGEALVNGQEKGR',
                                  6: 'MPQRFIVVTGGVLSGIGKGIFSASLARILKDSGVNVNILKIDPYLNVDAGTMNPNQHGEVFVTDDGYEADLDLGHYERFLGINVSRKNNITAGQIYYSVIKREREGKYLGSTVQIVPHVTSEIKDRIKTMDGDLLVIEIGGTVGDIEGEVFLEAVRELAFEIGREKFHFVHVTYVPYLRTTNEFKTKPTQQSVQLLRRIGIHPDTIIVRTEMPIDANSLFKVSLFSGVPRNRVINLPDASNVYEVPDVLHSLNLHKLIAKELDIDINDRFNWSYPKSFELLKIGIVGKYLGTDDAYKSIIESIYLSGAQKPIVIDAQELEDMTDEQIKNYLDDFDALIIPGGFGRRGIEGKIKAIKYARENKKPILGICLGMQLMAIEFARNVGKLEGANSTEFDENTPYPVVNMMESQKEVLNLGGTMRLGAQKTQIMKGTLLSRIYDGQEVVYERHRHRYEVDAEAFPQLFKNPGEEGYKLTISARSDFVEAVELDDHPFFVGIQYHPEYKSKVGKPHPIFKWLVKAAGGKIND',
                                  7: 'MTSLADLPVDVSPRHEGERIRSGDMYVELAGPKSFGAELFKVVDPDEIEPDKVEVIGPDIDEMEEGGRYPFAIYVKAAGEELEEDVEGVLERRIHEFCNYVEGFMHLNQRDQIWCRVSKNVTEKGFRLEHLGIALRELYKEEFGNVIDSVEVTIMTDEEKVEEFLEYARRVYKKRDERAKGLSEEDVNEFYVCLMCQSFAPTHVCVITPDRPSLCGSITWHDAKAAYKIDPEGPIFPIEKGECLDPEAGEYEGVNEAVKEHSQGTVERVYLHSCLEYPHTSCGCFQAVVFYIPEVDGFGIVDREYPGETPIGLPFSTMAGEASGGEQQPGFVGVSYGYMESDKFLQYDGGWERVVWMPKALKERMKHAIPDELYDKIATEEDATTVEELREFLEKVEHPVVERWAEEEEEEEEKAPEEEAPAEEPTMEVKELPIAPGGGLNVKIVLKNAKIYAEKVIIKRADREDKS',
                                  8: 'MGSADDRRFEVLRAIVADFVATKEPIGSKTLVERHNLGVSSATVRNDMAVLEAEGYITQPHTSSGRVPTEKGYREFVDRIDNVKPLSSSERRAILNFLESGVDLDDVLRRAVRLLAQLTRQVAIVQYPTLSTSSVRHLEVVALTPARLLLVVITDTGRVDQRIVELGDAIDEHELSKLRDMLGQAMEGKPLAQASIAVSDLASHLNGSDRLGDAVGRAATVLVETLVEHTEERLLLGGTANLTRNTADFGGSLRSVLEALEEQVVVLRLLAAQQEAGKVTVRIGHETEAEQMAGASVVSTAYGSSGKVYGGMGVVGPTRMDYPGTIANVAAVALYIGEVLGSR',
                                  9: 'MDNIRNFSIIAHIDHGKSTLADRIIQLCGGLSDREMEAQVLDSMDIEKERGITIKAQTAALSYKARDGKVYNLNLIDTPGHVDFSYEVSRSLSACEGALLVVDASQGVEAQTVANCYTAIELGVEVVPVLNKIDLPAADPDNAIQEIEDVIGIDAADATRCSAKTGEGVADVLEALIAKVPAPKGDPAAPLQALIIDSWFDNYVGVVMLVRVVNGTLRAKDKVLLMATGAQHLVEQVGVFSPKSVPRESLSAGQVGFVIAGIKELKAAKVGDTITHVAPRKAEAPLPGFKEVKPQVFAGLYPVEANQYEALRESLEKLKLNDASLQYEPEVSQALGFGFRCGFLGLLHMEIVQERLEREFDMDLITTAPTVVYQVQLRDGTMVQVENPAKMPADPSKIEAILEPIVTVNLYMPQEYVGAVITLCEQKRGSQINMSYHGRQVQLTYEIPMGEIVLDFFDRLKSVSRGYASMDYEFKEYRVSDVVKVDILINGDKVDALSIIVHRSNSTYRGREVAAKMREIIPRQMYDVAIQAAIGANVIARENVKALRKNVLAKCYGGDISRKKKLLEKQKEGKKRMKQVGTVEIPQEAFLAILRVEEK'},
                     'TCDB_ID': {0: '0', 1: '0', 2: '0', 3: '0', 4: '0', 5: '0', 6: '0', 7: '0', 8: '0', 9: '0'}}

        self.dataset = pd.DataFrame(test_data)
        # sequence = ReadSequence()  # create the object to process the sequencedsic
        # dataset = sequence.par_preprocessing_20AA(dataset, 'sequence')
        # self.dataset = ParDescritors(dataset, 'sequence')  # creating object to calculate descriptors

    def test_preprocessing20AA(self):
        protein = "ARNDCEQGHILKMFPSTWYVBZUOJX"
        res1 = 'ARNDCEQGHILKMFPSTWYVNQCKI'
        trans = ReadSequence()
        res = trans.get_preprocessing(protein)
        self.assertEqual(res,res1)

    def test_par_preprocessing20AA(self):
        res1 = self.dataset.shape
        trans = ReadSequence()
        res = trans.par_preprocessing(self.dataset, 'sequence')
        self.assertEqual(res.shape,res1)

    # def test_lenght(self):
    #     pass
    #
    # def test_charge(self):
    #     pass
    #
    # def test_charge_density(self):
    #     pass
    #
    # def test_formula(self):
    #     pass
    #
    # def test_bond(self):
    #     pass
    #
    # def test_mw(self):
    #     pass
    #
    # def test_gravy(self):
    #     pass
    #
    # def test_aromacity(self):
    #     pass
    #
    # def test_isoelectric_point(self):
    #     pass
    #
    # def test_instability_index(self):
    #     pass
    #
    # def test_sec_struct(self):
    #     pass
    #
    # def test_molar_extinction_coefficient(self):
    #     pass
    #
    # def test_flexibility(self):
    #     pass
    #
    # def test_aliphatic_index(self):
    #     pass
    #
    # def test_boman_index(self):
    #     pass
    #
    # def test_hydrophobic_ratio(self):
    #     pass

    ################## AMINO ACID COMPOSITION ##################

    # def test_aa_comp(self):
    #     pass
    #
    # def test_dp_comp(self):
    #     pass
    #
    # def test_tp_comp(self):
    #     pass
    #
    # ################## PSEUDO AMINO ACID COMPOSITION ##################
    #
    # def test_paac(self):
    #     pass
    #
    # def test_paac_p(self):
    #     pass
    #
    # def test_apaac(self):
    #     pass
    #
    # ################# AUTOCORRELATION DESCRIPTORS ##################
    #
    # def test_moreau_broto_auto(self):
    #     pass
    #
    # def test_moran_auto(self):
    #     pass
    #
    # def test_geary_auto(self):
    #     pass
    #
    # # ################# COMPOSITION, TRANSITION, DISTRIBUTION ##################
    #
    # def test_ctd(self):
    #     pass
    #
    # # ################# CONJOINT TRIAD ##################
    #
    # def test_conj_t(self):
    #     pass
    #
    # # #################  SEQUENCE ORDER  ##################
    #
    # def test_socn(self):
    #     pass
    #
    # def test_socn_p(self):
    #     pass
    #
    # def test_qso(self):
    #     pass
    #
    # def test_qso_p(self):
    #     pass
    #
    # # ################# BASE CLASS PEPTIDE DESCRIPTOR ##################
    #
    # def descriptor_calculate_moment(self):
    #     pass
    #
    # def descriptor_calculate_global(self):
    #     pass
    #
    # def descriptor_calculate_profile(self):
    #     pass
    #
    # def descriptor_calculate_arc(self):
    #     pass
    #
    # def descriptor_calculate_autocorr(self):
    #     pass
    #
    # def descriptor_calculate_crosscorr(self):
    #     pass
    #
    # def test_all_physicochemical(self):
    #     pass
    #
    # def test_all_aac(self):
    #     pass
    #
    # def test_all_paac(self):
    #     pass
    #
    # def test_all_sequenceorder(self):
    #     pass
    #
    # def test_all_correlation(self):
    #     pass
    #
    # def test_all_base_class(self):
    #     pass
    #
    # def test_all(self):
    #     pass
