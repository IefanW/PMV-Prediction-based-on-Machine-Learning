import joblib
import numpy as np
import time

sensation = "Neutral"

def run():
    dt_model = joblib.load("trained_model.pkl")
    print("-----Shell application for PMV prediction-----\n"
          "----------------------------------------------")
    print("--What would you like to do? (A/B)\n"
              "--A. Predicting the PMV value according to the data you have.\n"
              "--B. Quit")
    while True:
        input_command = input("--").strip()
        if input_command != "":
            if input_command == "A":
                print("\n--Input the features you have/ cancel (input 'back'):")
                print("--Outdoor temperature,Indoor temperature,Related humidity,CO2")
                while True:
                    features = input("--Format: x1,x2,x3,x4\n")
                    if features != "" or features!= None:
                        features_list = features.strip().split(",")
                        if len(features_list) != 4:
                            if features == "back":
                                break
                            print("The format is wrong!!! Try again!!!")
                        else:
                            features_input = np.array([features_list])
                            print("Classifying......")
                            time.sleep(2)
                            result = dt_model.predict(features_input)
                            prediction = result[0]
                            print("The predicted PMV value is: " + str(prediction))
                            prediction = int(float(prediction))

                            global sensation
                            if prediction == -3:
                                sensation = "Cold"
                            elif prediction == -2:
                                sensation = "Cool"
                            elif prediction == -1:
                                sensation = "Slightly Cool"
                            elif prediction == 1:
                                sensation = "Slightly Warm"
                            elif prediction == 2:
                                sensation = "Warm"
                            elif prediction == 3:
                                sensation = "Hot"

                            print("The corresponding sensation is: " + sensation + "\n")
                            break
                    else:
                        print("Input is empty!!")

            elif input_command == "B":
                print("Thanks for using\n"
                      "The application is shutting down")
                time.sleep(3)
                break
            else:
                print("Choose the right command.")
        else:
            print("Input is empty!!!")

if __name__ == '__main__':
    run()