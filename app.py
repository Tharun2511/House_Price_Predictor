from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import array
app = Flask(__name__)

model=pickle.load(open('House-price-prediction.pkl','rb'))
scaler = joblib.load("Standard_Scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        variables=['MSSubClass', 'LotFrontage', 'LotArea', 'Utilities', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual',
       'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',
       'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'MoSold', 'YrSold',
       'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
       'Street_Pave', 'Alley_NA', 'Alley_Pave', 'LotShape_IR2', 'LotShape_IR3',
       'LotShape_Reg', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl',
       'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3',
       'LotConfig_Inside', 'LandSlope_Mod', 'LandSlope_Sev',
       'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide',
       'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor',
       'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
       'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes',
       'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge',
       'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU',
       'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst',
       'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker',
       'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA',
       'Condition1_PosN','Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe',
       'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm',
       'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe',
       'Condition2_RRAn', 'Condition2_RRNn', 'BldgType_2fmCon',
       'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE',
       'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin',
       'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer',
       'HouseStyle_SLvl', 'RoofStyle_Gable', 'RoofStyle_Gambrel',
       'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed',
       'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal',
       'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake',
       'RoofMatl_WdShngl', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm',
       'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd',
       'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd',
       'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco',
       'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing',
       'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace',
       'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard',
       'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other',
       'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco',
       'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng',
       'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
       'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab',
       'Foundation_Stone', 'Foundation_Wood', 'Heating_GasA', 'Heating_GasW',
       'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'CentralAir_Y',
       'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix',
       'Electrical_SBrkr', 'GarageType_Attchd', 'GarageType_Basment',
       'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd',
       'GarageType_NA', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA',
       'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed',
       'MiscFeature_TenC', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD',
       'SaleType_ConLI', 'SaleType_ConLw','SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_AdjLand',
       'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
       'SaleCondition_Partial']
        app.logger.info(variables)
        values=[0]*207
        LotFrontage=int(request.form["LotFrontage"])
        values[variables.index("LotFrontage")]=LotFrontage
        LotArea=int(request.form["LotArea"])
        values[variables.index("LotArea")]=LotArea
        Utilities=int(request.form["Utilities"])
        values[variables.index("Utilities")]=Utilities
        OverallQual=int(request.form["OverallQual"])
        values[variables.index("OverallQual")]=OverallQual
        OverallCond=int(request.form["OverallCond"])
        values[variables.index("OverallCond")]=OverallCond
        MasVnrArea=int(request.form["MasVnrArea"])
        values[variables.index("MasVnrArea")]=MasVnrArea
        ExterQual=int(request.form["ExterQual"])-1
        values[variables.index("ExterQual")]=ExterQual
        ExterCond=int(request.form["ExterCond"])
        values[variables.index("ExterCond")]=ExterCond
        BsmtQual=int(request.form["BsmtQual"])
        values[variables.index("BsmtQual")]=BsmtQual
        BsmtCond=int(request.form["BsmtCond"])
        values[variables.index("BsmtCond")]=BsmtCond
        BsmtExposure=int(request.form["BsmtExposure"])
        values[variables.index("BsmtExposure")]=BsmtExposure
        BsmtFinType1=int(request.form["BsmtFinType1"])
        values[variables.index("BsmtFinType1")]=BsmtFinType1
        BsmtFinSF1=int(request.form["BsmtFinSF1"])
        values[variables.index("BsmtFinSF1")]=BsmtFinSF1
        BsmtFinType2=1
        values[variables.index("BsmtFinType2")]=BsmtFinType2
        BsmtFinSF2=0
        values[variables.index("BsmtFinSF2")]=BsmtFinSF2
        BsmtUnfSF=int(request.form["BsmtUnfSF"])
        values[variables.index("BsmtUnfSF")]=BsmtUnfSF
        TotalBsmtSF=int(request.form["TotalBsmtSF"])
        values[variables.index("TotalBsmtSF")]=TotalBsmtSF
        HeatingQC=int(request.form["HeatingQC"])-1
        values[variables.index("HeatingQC")]=HeatingQC
        OstFlrSF=int(request.form["1stFlrSF"])
        values[variables.index("1stFlrSF")]=OstFlrSF
        TndFlrSF=int(request.form["2ndFlrSF"])
        values[variables.index("2ndFlrSF")]=TndFlrSF
        LowQualFinSF=int(request.form["LowQualFinSF"])
        values[variables.index("LowQualFinSF")]=LowQualFinSF
        GrLivArea=int(request.form["GrLivArea"])
        values[variables.index("GrLivArea")]=GrLivArea
        BsmtFullBath=int(request.form["BsmtFullBath"])
        values[variables.index("BsmtFullBath")]=BsmtFullBath
        BsmtHalfBath=int(request.form["BsmtHalfBath"])
        values[variables.index("BsmtHalfBath")]=BsmtHalfBath
        FullBath=int(request.form["FullBath"])
        values[variables.index("FullBath")]=FullBath
        HalfBath=int(request.form["HalfBath"])
        values[variables.index("HalfBath")]=HalfBath
        BedroomAbvGr=int(request.form["BedroomAbvGr"])
        values[variables.index("BedroomAbvGr")]=BedroomAbvGr
        KitchenAbvGr=int(request.form["KitchenAbvGr"])
        values[variables.index("KitchenAbvGr")]=KitchenAbvGr
        KitchenQual=int(request.form["KitchenQual"])-1
        values[variables.index("KitchenQual")]=KitchenQual
        TotRmsAbvGrd=int(request.form["TotRmsAbvGrd"])
        values[variables.index("TotRmsAbvGrd")]=TotRmsAbvGrd
        Functional=int(request.form["Functional"])
        values[variables.index("Functional")]=Functional
        Fireplaces=int(request.form["Fireplaces"])
        values[variables.index("Fireplaces")]=Fireplaces
        FireplaceQu=int(request.form["FireplaceQu"])
        values[variables.index("FireplaceQu")]=FireplaceQu
        GarageFinish=int(request.form["GarageFinish"])
        values[variables.index("GarageFinish")]=GarageFinish
        GarageCars=int(request.form["GarageCars"])
        values[variables.index("GarageCars")]=GarageCars
        GarageArea=int(request.form["GarageArea"])
        values[variables.index("GarageArea")]=GarageArea
        GarageQual=int(request.form["GarageQual"])
        values[variables.index("GarageQual")]=GarageQual
        GarageCond=int(request.form["GarageCond"])
        values[variables.index("GarageCond")]=GarageCond
        PavedDrive=int(request.form["PavedDrive"])
        values[variables.index("PavedDrive")]=PavedDrive
        WoodDeckSF=int(request.form["WoodDeckSF"])
        values[variables.index("WoodDeckSF")]=WoodDeckSF
        OpenPorchSF=int(request.form["OpenPorchSF"])
        values[variables.index("OpenPorchSF")]=OpenPorchSF
        EnclosedPorch=int(request.form["EnclosedPorch"])
        values[variables.index("EnclosedPorch")]=EnclosedPorch
        SsnPorch=int(request.form["3SsnPorch"])
        values[variables.index("3SsnPorch")]=SsnPorch
        ScreenPorch=int(request.form["ScreenPorch"])
        values[variables.index("ScreenPorch")]=ScreenPorch
        PoolArea=int(request.form["PoolArea"])
        values[variables.index("PoolArea")]=PoolArea
        PoolQC=int(request.form["PoolQC"])-1
        values[variables.index("PoolQC")]=PoolQC
        MiscVal=int(request.form["MiscVal"])
        values[variables.index("MiscVal")]=1
        MSSubClass=str(request.form["MSSubClass"])
        values[variables.index("MSSubClass")]=MSSubClass
        MSZoning="MSZoning_"+str(request.form["MSZoning"])
        values[variables.index(MSZoning)]=1
        Street="Street_"+str(request.form["Street"])
        if(Street!="Street_Grvl"):
            values[variables.index(Street)]=1
        Alley="Alley_"+str(request.form["Alley"])
        values[variables.index(Alley)]=1
        LotShape="LotShape_"+str(request.form["LotShape"])
        values[variables.index(LotShape)]=1
        LandContour="LandContour_"+str(request.form["LandContour"])
        values[variables.index(LandContour)]=1
        LotConfig="LotConfig_"+str(request.form["LotConfig"])
        values[variables.index(LotConfig)]=1
        LandSlope="LandSlope_"+str(request.form["LandSlope"])
        values[variables.index(LandSlope)]=1
        Neighborhood="Neighborhood_"+str(request.form["Neighborhood"])
        values[variables.index(Neighborhood)]=1
        Condition1="Condition1_"+str(request.form["Condition1"])
        values[variables.index(Condition1)]=1
        Condition2="Condition2_Norm"
        values[variables.index(Condition2)]=1
        BldgType="BldgType_"+str(request.form["BldgType"])
        values[variables.index(BldgType)]=1
        YearBuilt=int(str(request.form["YearBuilt"]))
        values[variables.index("YearBuilt")]=YearBuilt
        RoofStyle="RoofStyle_"+str(request.form["RoofStyle"])
        values[variables.index(RoofStyle)]=1
        RoofMatl="RoofMatl_"+str(request.form["RoofMatl"])
        values[variables.index(RoofMatl)]=1
        Exterior1st="Exterior1st_"+str(request.form["Exterior1st"])
        values[variables.index(Exterior1st)]=1
        Exterior2nd="Exterior2nd_VinylSd"
        values[variables.index(Exterior2nd)]=1
        MasVnrType="MasVnrType_"+str(request.form["MasVnrType"])
        values[variables.index(MasVnrType)]=1
        Foundation="Foundation_"+str(request.form["Foundation"])
        values[variables.index(Foundation)]=1
        Heating="Heating_"+str(request.form["Heating"])
        values[variables.index(Heating)]=1
        CentralAir="CentralAir_"+str(request.form["CentralAir"])
        if(CentralAir=="CentralAir_Y"):
            values[variables.index("CentralAir_Y")]=1
        Electrical="Electrical_"+str(request.form["Electrical"])
        values[variables.index(Electrical)]=1
        GarageType="GarageType_"+str(request.form["GarageType"])
        values[variables.index(GarageType)]=1
        Fence="Fence_"+str(request.form["Fence"])
        values[variables.index(Fence)]=1
        MiscFeature="MiscFeature_"+str(request.form["MiscFeature"])
        values[variables.index(MiscFeature)]=1
        SaleCondition="SaleCondition_"+str(request.form["SaleCondition"])
        values[variables.index(SaleCondition)]=1
        values=np.reshape(values,(1,-1))
        values = scaler.transform(values)
        features=np.array(values,dtype=object)
        print("Lohggerbkjb")
        with open('House-price-prediction.pkl', 'rb') as file:
            model = pickle.load(file)
            prediction=model.predict(features)
            return render_template('index.html', prediction_text=f'{prediction}')
    else:
        return render_template('index.html')
    
if __name__=="__main__":
    app.run(debug=True)
