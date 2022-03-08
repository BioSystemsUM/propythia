"""
##############################################################################

File containing tests functions to check if all functions from descriptors module are properly working

Authors: Ana Marta Sequeira, Miguel Barros

Date: 06/2019 ALTERED 03/2022

Email:

##############################################################################
"""

from unittest import TestCase
from propythia.Par_descriptores import ParDescritors
from propythia.sequence import ReadSequence
from propythia.descriptors import Descriptor
import timeit
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

        dataset = pd.DataFrame(test_data)
        sequence = ReadSequence()  # create the object to process the sequencedsic
        self.dataset = sequence.par_preprocessing_20AA(dataset, 'sequence')

    def test_all_physicochemical(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_physicochemical()
        self.assertEqual(res1, data.shape[0])

    def test_all_aac(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_aac()
        self.assertEqual(res1, data.shape[0])

    def test_all_paac(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_paac()
        self.assertEqual(res1, data.shape[0])

    def test_all_sequenceorder(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_sequenceorder()
        self.assertEqual(res1, data.shape[0])

    def test_all_correlation(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_correlation()
        self.assertEqual(res1, data.shape[0])

    def test_all_base_class(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all_base_class()
        self.assertEqual(res1, data.shape[0])

    def test_all(self):
        res1 = self.dataset.shape[0]
        desc = ParDescritors(self.dataset, 'sequence')
        data = desc.par_all()
        self.assertEqual(res1, data.shape[0])

    def test_descriptors(self):
        sequence = ReadSequence()  # create the object to read sequence

        ps = sequence.get_protein_sequence_from_id('P48039')  # from uniprot id
        protein = Descriptor(ps)  # creating object to calculate descriptors

        ps_string = sequence.read_protein_sequence(
            "MQGNGSALPNASQPVLRGDGARPSWLASALACVLIFTIVVDILGNLLVILSVYRNKKLRN")  # from string
        protein2 = Descriptor(ps_string)

        print('tests protein1')
        start = timeit.default_timer()
        test1 = protein.get_all()  # all except tripeptide and binaries representations
        test1_2 = protein.adaptable([1, 2, 23])  # bin aa and bin properties and aminoacid composition

        print(test1)
        print(len(test1))
        print(test1_2)
        print(len(test1_2))
        stop = timeit.default_timer()
        print('Time: ', stop - start)  # 8''/350 aa

        print('tests protein2')
        start = timeit.default_timer()
        test2 = protein2.get_all()
        test2_2 = protein2.adaptable([1, 2, 23])
        print(test2)
        print(len(test2))
        print(test2_2)
        print(len(test2_2))
        stop = timeit.default_timer()
        print('Time: ', stop - start)  # <4''/60 aa

    if __name__ == "__main__":
        test_descriptors()
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
