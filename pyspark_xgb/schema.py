from pyspark.sql.types import StructField, StructType, IntegerType, DoubleType, FloatType, StringType


def get_btrain_schema():
    ''' binary '''
    return StructType([
        StructField('Age', IntegerType(), True),
        StructField('Attrition', DoubleType(), True),
        StructField('BusinessTravel', IntegerType(), True),
        StructField('DailyRate', IntegerType(), True),
        StructField('Department', IntegerType(), True),
        StructField('DistanceFromHome', IntegerType(), True),
        StructField('Education', IntegerType(), True),
        StructField('EducationField', IntegerType(), True),
        StructField('EnvironmentSatisfaction', IntegerType(), True),
        StructField('Gender', IntegerType(), True),
        StructField('HourlyRate', IntegerType(), True),
        StructField('JobInvolvement', IntegerType(), True),
        StructField('JobLevel', IntegerType(), True),
        StructField('JobSatisfaction', IntegerType(), True),
        StructField('MaritalStatus', IntegerType(), True),
        StructField('MonthlyIncome', IntegerType(), True),
        StructField('MonthlyRate', IntegerType(), True),
        StructField('NumCompaniesWorked', IntegerType(), True),
        StructField('OverTime', IntegerType(), True),
        StructField('PercentSalaryHike', IntegerType(), True),
        StructField('PerformanceRating', IntegerType(), True),
        StructField('RelationshipSatisfaction', IntegerType(), True),
        StructField('StockOptionLevel', IntegerType(), True),
        StructField('TotalWorkingYears', IntegerType(), True),
        StructField('TrainingTimesLastYear', IntegerType(), True),
        StructField('WorkLifeBalance', IntegerType(), True),
        StructField('YearsAtCompany', IntegerType(), True),
        StructField('YearsInCurrentRole', IntegerType(), True),
        StructField('YearsSinceLastPromotion', IntegerType(), True),
        StructField('YearsWithCurrManager', IntegerType(), True)])


def get_bpred_schema():
    ''' binary '''
    return StructType([
        StructField('Age', IntegerType(), True),
        StructField('BusinessTravel', IntegerType(), True),
        StructField('DailyRate', IntegerType(), True),
        StructField('Department', IntegerType(), True),
        StructField('DistanceFromHome', IntegerType(), True),
        StructField('Education', IntegerType(), True),
        StructField('EducationField', IntegerType(), True),
        StructField('EnvironmentSatisfaction', IntegerType(), True),
        StructField('Gender', IntegerType(), True),
        StructField('HourlyRate', IntegerType(), True),
        StructField('JobInvolvement', IntegerType(), True),
        StructField('JobLevel', IntegerType(), True),
        StructField('JobSatisfaction', IntegerType(), True),
        StructField('MaritalStatus', IntegerType(), True),
        StructField('MonthlyIncome', IntegerType(), True),
        StructField('MonthlyRate', IntegerType(), True),
        StructField('NumCompaniesWorked', IntegerType(), True),
        StructField('OverTime', IntegerType(), True),
        StructField('PercentSalaryHike', IntegerType(), True),
        StructField('PerformanceRating', IntegerType(), True),
        StructField('RelationshipSatisfaction', IntegerType(), True),
        StructField('StockOptionLevel', IntegerType(), True),
        StructField('TotalWorkingYears', IntegerType(), True),
        StructField('TrainingTimesLastYear', IntegerType(), True),
        StructField('WorkLifeBalance', IntegerType(), True),
        StructField('YearsAtCompany', IntegerType(), True),
        StructField('YearsInCurrentRole', IntegerType(), True),
        StructField('YearsSinceLastPromotion', IntegerType(), True),
        StructField('YearsWithCurrManager', IntegerType(), True)])


def get_mtrain_schema():
    return StructType([
        StructField('sepal_length', FloatType(), False),
        StructField('sepal_width', FloatType(), True),
        StructField('petal_length', FloatType(), False),
        StructField('petal_width', FloatType(), False),
        StructField('class', StringType(), False),
        StructField('label', DoubleType(), False)])


def get_mtest_schema():
    return StructType([
        StructField('sepal_length', FloatType(), False),
        StructField('sepal_width', FloatType(), True),
        StructField('petal_length', FloatType(), False),
        StructField('petal_width', FloatType(), False)])
