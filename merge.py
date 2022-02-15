import random

import numpy
import pandas

random.seed(1126)

demographics: pandas.DataFrame = pandas.read_csv("./data/demographics.csv")
satisfaction: pandas.DataFrame = pandas.read_csv("./data/satisfaction.csv")
services: pandas.DataFrame = pandas.read_csv("./data/services.csv")
status: pandas.DataFrame = pandas.read_csv("./data/status.csv")

location: pandas.DataFrame = pandas.read_csv("./data/location.csv")
population: pandas.DataFrame = pandas.read_csv("./data/population.csv")
# ID is not useful
population.drop("ID", axis=1, inplace=True)


def merge(IDs: pandas.DataFrame, testdata: bool) -> pandas.DataFrame:
    """
        merge all the given datasets, including `demographics.csv`, `location.csv`,
        `population.csv`, `satisfaction.csv`, `services.csv` and `status.csv`, with the
        given `IDs`.

        Parameters
        ----------
        `IDs`      : the IDs of the customers

        `testdata` : indicate the IDs are test data or not
    """
    # merge all the data
    data: pandas.DataFrame
    data = pandas.merge(IDs, demographics, how="left",
                        on="Customer ID", validate="1:1")
    data = pandas.merge(data, location, how="left",
                        on="Customer ID", validate="1:1")
    data = pandas.merge(data, satisfaction, how="left",
                        on="Customer ID", validate="1:1")
    data = pandas.merge(data, services, how="left",
                        on="Customer ID", validate="1:1")
    data = pandas.merge(data, status, how="left",
                        on="Customer ID", validate="1:1")

    def fill_nan_with_sampling(field: str | list[str]) -> None:
        if type(field) is str:
            field = [field]
        for f in field:
            empty = data[f].isna()
            data.loc[empty, f] = random.choices(
                data.loc[~empty, f].values, k=empty.sum())

    data.drop("Customer ID", axis=1, inplace=True)
    data.drop("Count_x", axis=1, inplace=True)
    fill_nan_with_sampling("Gender")

    # fill Age according to Under 30
    under30 = data["Age"] < 30
    least30 = data["Age"] >= 30
    filling = (data["Under 30"] == "Yes") & data["Age"].isna()
    data.loc[filling, "Age"] = random.choices(
        data.loc[under30, "Age"].values, k=filling.sum())
    filling = (data["Under 30"] == "No") & data["Age"].isna()
    data.loc[filling, "Age"] = random.choices(
        data.loc[least30, "Age"].values, k=filling.sum())
    data.drop("Under 30", axis=1, inplace=True)
    fill_nan_with_sampling("Age")

    fill_nan_with_sampling(["Senior Citizen", "Married"])

    # fill Number of Dependents according to Dependents
    has_dependents = data["Number of Dependents"] > 0
    filling = ((data["Dependents"] == "Yes") &
               data["Number of Dependents"].isna())
    data.loc[filling, "Number of Dependents"] = random.choices(
        data.loc[has_dependents, "Number of Dependents"].values,
        k=filling.sum()
    )
    data.loc[data["Dependents"] == "No", "Number of Dependents"] = 0
    data.drop("Dependents", axis=1, inplace=True)
    fill_nan_with_sampling("Number of Dependents")

    data.drop("Count_y", axis=1, inplace=True)
    data.drop("Country", axis=1, inplace=True)
    data.drop("State", axis=1, inplace=True)
    fill_nan_with_sampling(["City", "Zip Code"])

    # update Latitude and Longitude from Lat Long because some of them are empty
    latlong = data["Lat Long"].str.split(pat=", ", expand=True).astype(float)
    data["Latitude"].update(latlong[0])
    data["Longitude"].update(latlong[1])
    # Lat Long is duplicated with Latitude and Longitude
    data.drop("Lat Long", axis=1, inplace=True)
    fill_nan_with_sampling(["Latitude", "Longitude"])

    # merge Population
    data = pandas.merge(data, population, how="left",
                        on="Zip Code", validate="m:1")
    data_population = data.pop("Population")
    data.insert(9, "Population", data_population)

    fill_nan_with_sampling("Satisfaction Score")
    data.drop("Count", axis=1, inplace=True)
    data.drop("Quarter", axis=1, inplace=True)
    fill_nan_with_sampling(["Referred a Friend", "Number of Referrals",
                           "Tenure in Months", "Offer", "Phone Service",
                            "Avg Monthly Long Distance Charges", "Multiple Lines"])

    # fill Internet Type according to Internet Service
    data.loc[data["Internet Service"] == "No", "Internet Type"] = "None"
    data.drop("Internet Service", axis=1, inplace=True)
    fill_nan_with_sampling("Internet Type")
    data.loc[data["Internet Type"] == "None", "Avg Monthly GB Download"] = 0
    data.loc[data["Internet Type"] == "None",
             ["Online Security", "Online Backup", "Device Protection Plan",
             "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music",
              "Unlimited Data"]] = "No"

    fill_nan_with_sampling(["Avg Monthly GB Download", "Online Security",
                           "Online Backup", "Device Protection Plan",
                            "Premium Tech Support", "Streaming TV",
                            "Streaming Movies", "Streaming Music", "Unlimited Data",
                            "Contract", "Paperless Billing", "Payment Method",
                            "Monthly Charge", "Total Charges", "Total Refunds",
                            "Total Extra Data Charges", "Total Long Distance Charges",
                            "Total Revenue"])

    # replace empty string with nan
    data.replace("", numpy.nan, inplace=True)
    # delete the data if all of the fields are empty
    data.dropna(axis=0, how="all", inplace=True)
    if not testdata:
        # filter unknown churn category
        data = data[data["Churn Category"].notna()]

    def convert_yes_no_to_int(field: str | list[str]):
        if type(field) is str:
            field = [field]
        for f in field:
            data[f] = data[f].apply(lambda y: 1 if y == "Yes" else -1)

    # convert string to numerical values
    data["Gender"] = data["Gender"].apply(lambda g: 1 if g == "Male" else -1)
    convert_yes_no_to_int(["Senior Citizen", "Married", "Referred a Friend",
                           "Phone Service", "Multiple Lines", "Online Security",
                           "Online Backup", "Device Protection Plan", "Premium Tech Support",
                           "Streaming TV", "Streaming Movies", "Streaming Music",
                           "Unlimited Data", "Paperless Billing"])
    if testdata:
        data.drop("Churn Category", axis=1, inplace=True)
    else:
        data.replace({"Churn Category": {"No Churn": 0, "Competitor": 1,
                     "Dissatisfaction": 2, "Attitude": 3, "Price": 4, "Other": 5}}, inplace=True)
    return data


Test_IDs = pandas.read_csv("./data/Test_IDs.csv")
Train_IDs = pandas.read_csv("./data/Train_IDs.csv")

train_data, test_data = merge(Train_IDs, False), merge(Test_IDs, True)

train_data.to_csv("./train.csv", index=False)
test_data.to_csv("./test.csv", index=False)