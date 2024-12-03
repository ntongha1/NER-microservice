import re
from pyspark.sql import functions as F, types as T
from pyspark.sql.types import StringType, IntegerType, StructField, StructType, MapType, ArrayType, FloatType
import pandas as pd

dataTypeModifier = StructType([
            StructField('annotatorType', StringType(), True),
            StructField('begin', IntegerType(), False),
            StructField('end', IntegerType(), False),
            StructField('result', StringType(), True),
            StructField('metadata', MapType(StringType(), StringType()), True),
            StructField('embeddings', ArrayType(FloatType(), False), True)
        ])

dataType = StructType([
    StructField('annotatorType', StringType(), False),
    StructField('begin', IntegerType(), False),
    StructField('end', IntegerType(), False),
    StructField('result', StringType(), False),
    StructField('metadata', MapType(StringType(), StringType()), False),
    StructField('embeddings', ArrayType(FloatType()), False)
])


target_chunks = ['Disease_Syndrome_Disorder', 'Diabetes', 'Oncological', 'Modifier', 'Procedure'
'Heart_Disease', 'Kidney_Disease', 'Substance', 'Cerebrovascular_Disease','Psychological_Condition']


# vctrz = F.udf(agg_embs, ArrayType(dataType))

# chunk_udf = F.udf(chunk_filter, ArrayType(dataTypeModifier))


# text clearning
# clean_text = lambda s: re.sub(r'†', ' ', s)
# clean_new_line = lambda s: re.sub(r'\n', ' ', s)


def agg_embs(embs):
    l = len(embs)
    text_list = []
    for i in range(l):
        text_list.append(["feature_vector", 
        embs[i].begin, 
        embs[i].end, 
        embs[i].result,
        embs[i].metadata,
        embs[i].embeddings]) 
    return text_list


def chunk_filter (chunk_embedding, clf_result):
    filter_list = []
    for i in range(len(chunk_embedding)):
        if clf_result[i].result == '1.0':
            filter_list.append(chunk_embedding[i])           
    return filter_list


def remove_html_tags(text):
    clean1 = re.compile('<.*?>')
    clean2 = re.compile('\n')
    clean3 = re.compile('†')
    new_text = re.sub(clean1, '', text)
    new_text = re.sub(clean2, ' ', new_text)
    new_text = re.sub(clean3, ' ', new_text)
    return new_text


# metadata filter
def get_icd10_codes (output, path, distance, confidence=0.1):
    print('get icd10 codes', flush=True)
    codes_df = pd.read_csv(path)
    code_list = codes_df[(~codes_df['Diagnosis\nCode'].isnull()) & (codes_df['Type'] == 'Commercial')]['Diagnosis\nCode'].tolist()

    df = output.select("text_id", F.explode(F.arrays_zip("ner_chunk_document.result","ner_chunk_document.begin",
                                              "ner_chunk_document.end","ner_chunk_document.metadata", 
                                              "icd_resolution.result","icd_resolution.metadata"
                                              )).alias("cols")) \
        .select("text_id", F.expr("cols['0']").alias("chunk"),
                F.expr("cols['1']").alias("begin"),
                F.expr("cols['2']").alias("end"),
                F.expr("cols['4']").alias("pre_code"),
                F.expr("cols['5'].resolved_text").alias("resolved_text"), 
                F.expr("filter(arrays_zip(split(cols['5']['all_k_results'],':::'),split(cols['5']['all_k_confidences'],':::'),split(cols['5']['all_k_cosine_distances'],':::'),split(cols['5']['all_k_resolutions'],':::')),x -> x['0'] in ('" + '\',\''.join(code_list) +"'))").alias('icd10_res')) \
               .select("text_id", "chunk", "begin", "end", "resolved_text",
                "pre_code",
                F.expr("icd10_res[0]['3']").alias("description"),
                F.expr("icd10_res[0]['2']").alias("distance"),
                F.expr("icd10_res[0]['1']").alias("confidence"),
                F.expr("icd10_res[0]['0']").alias("code")
               ).dropna()
    df = df[df['distance'] <= distance]
    df = df[df['confidence'] >= confidence]    
    return df


def insulin_codes(output):
    print('find insulin codes', flush=True)
    df = output.select("text_id",F.col('drug_match')[0].alias("cols")) \
            .select("text_id",
                    F.expr("cols.result").alias("chunk"),
                    F.expr("cols.begin").alias("begin"),
                    F.expr("cols.end").alias("end"))\
            .withColumn("resolved_text", F.col("chunk"))\
            .withColumn("pre_code", F.lit("Z794"))\
            .withColumn("description", F.lit("Long term (current) use of insulin"))\
            .withColumn("distance", F.lit(0.1000))\
            .withColumn("confidence", F.lit('0.9000'))\
            .withColumn("code", F.lit("Z794")).dropna()  
    return df


def insulin_check(df):
    dm_list = ['E0800','E0801','E0810','E0811','E0821','E0822','E0829','E08311','E08319',
            'E083211','E083212','E083213','E083219','E083291','E083292','E083293',
            'E083299','E083311','E083312','E083313','E083319','E083391','E083392',
            'E083393','E083399','E083411','E083412','E083413','E083419','E083491',
            'E083492','E083493','E083499','E083511','E083512','E083513','E083519',
            'E083521','E083522','E083523','E083529','E083531','E083532','E083533',
            'E083539','E083541','E083542','E083543','E083549','E083551','E083552',
            'E083553','E083559','E083591','E083592','E083593','E083599','E0836',
            'E0837X1','E0837X2','E0837X3','E0837X9','E0839','E0840','E0841','E0842',
            'E0843','E0844','E0849','E0851','E0852','E0859','E08610','E08618','E08620',
            'E08621','E08622','E08628','E08630','E08638','E08641','E08649','E0865',
            'E0869','E088','E089','E0900','E0901','E0910','E0911','E0921','E0922',
            'E0929','E09311','E09319','E093211','E093212','E093213','E093219','E093291',
            'E093292','E093293','E093299','E093311','E093312','E093313','E093319',
            'E093391','E093392','E093393','E093399','E093411','E093412','E093413',
            'E093419','E093491','E093492','E093493','E093499','E093511','E093512',
            'E093513','E093519','E093521','E093522','E093523','E093529','E093531',
            'E093532','E093533','E093539','E093541','E093542','E093543','E093549',
            'E093551','E093552','E093553','E093559','E093591','E093592','E093593',
            'E093599','E0936','E0937X1','E0937X2','E0937X3','E0937X9','E0939','E0940',
            'E0941','E0942','E0943','E0944','E0949','E0951','E0952','E0959','E09610',
            'E09618','E09620','E09621','E09622','E09628','E09630','E09638','E09641',
            'E09649','E0965','E0969','E098','E099','E1100','E1101','E1110','E1111',
            'E1121','E1122','E1129','E11311','E11319','E113211','E113212','E113213',
            'E113219','E113291','E113292','E113293','E113299','E113311','E113312',
            'E113313','E113319','E113391','E113392','E113393','E113399','E113411',
            'E113412','E113413','E113419','E113491','E113492','E113493','E113499',
            'E113511','E113512','E113513','E113519','E113521','E113522','E113523',
            'E113529','E113531','E113532','E113533','E113539','E113541','E113542',
            'E113543','E113549','E113551','E113552','E113553','E113559','E113591',
            'E113592','E113593','E113599','E1136','E1137X1','E1137X2','E1137X3',
            'E1137X9','E1139','E1140','E1141','E1142','E1143','E1144','E1149','E1151',
            'E1152','E1159','E11610','E11618','E11620','E11621','E11622','E11628',
            'E11630','E11638','E11641','E11649','E1165','E1169','E118','E119','E1300',
            'E1301','E1310','E1311','E1321','E1322','E1329','E13311','E13319','E133211',
            'E133212','E133213','E133219','E133291','E133292','E133293','E133299',
            'E133311','E133312','E133313','E133319','E133391','E133392','E133393',
            'E133399','E133411','E133412','E133413','E133419','E133491','E133492',
            'E133493','E133499','E133511','E133512','E133513','E133519','E133521',
            'E133522','E133523','E133529','E133531','E133532','E133533','E133539',
            'E133541','E133542','E133543','E133549','E133551','E133552','E133553',
            'E133559','E133591','E133592','E133593','E133599','E1336','E1337X1',
            'E1337X2','E1337X3','E1337X9','E1339','E1340','E1341','E1342','E1343',
            'E1344','E1349','E1351','E1352','E1359','E13610','E13618','E13620','E13621',
            'E13622','E13628','E13630','E13638','E13641','E13649','E1365','E1369','E138',
            'E139']

    if df.filter((F.col('code') == 'Z794')).count() ==1 and df.filter(F.col("code").isin(dm_list)).count() ==2:
        df = df.filter((F.col('code') != 'Z794'))
        return df
    else: 
        return df


# get icd code from the output of the pipeline
def get_icd_mapping_codes_for_pipeline_output(icd_extract, year=2018):
    print('get insulin and icd mapping codes', flush=True)

    # find insulin codes
    insulin = insulin_codes(icd_extract)

    if year == 2016: FY = './icd_mapping/Commercial_Model_Mappings_FY16_Oct_2015_Sep_2016.csv'
    if year == 2017: FY = './icd_mapping/Commercial_Model_Mappings_FY17_Oct_2016_Sep_2017.csv'
    if year == 2018: FY = './icd_mapping/Commercial_Model_Mappings_FY18_Oct_2017_Sep_2018.csv'
    if year == 2019: FY = './icd_mapping/Commercial_Model_Mappings_FY19_Oct_2018_Sep_2019.csv'
    if year == 2020: FY = './icd_mapping/Commercial_Model_Mappings_FY20_Oct_2019_Sep_2020.csv'
    if year == 2021: FY = './icd_mapping/Commercial_Model_Mappings_FY21_Oct_2020_Sep_2021.csv'
    
    icd_output = get_icd10_codes(icd_extract, FY, 8, 0).union(insulin)
    icd_output.cache()

    return icd_output


def get_comercial_list(path, icd_output):
    print('get_ commercial list', flush=True)
    maptest = pd.read_csv(path)
    maptest = maptest[["Diagnosis\nCode","Code Description"]].dropna()

    out_final = pd.merge(icd_output, maptest, left_on=['code'], right_on = ['Diagnosis\nCode'])
    code_list = out_final[["chunk","code","begin","end","Code Description","distance","confidence"]].drop_duplicates().reset_index()
    return code_list
