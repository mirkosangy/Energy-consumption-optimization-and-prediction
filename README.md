# Energy-consumption-optimization-and-prediction
Thanks to the progress of the information and communication technologies, in the last years the number of Smart Cities and IoT devices has greatly increased, bringing massive changes in many sectors. As far as the building design is concerned, ICT can be very helpful in the optimization, prediction and visualization of energy consumptions. Turning a traditional building into a smart building using smart technologies has become more and more popular. Reducing the energy cost and being more environment-friendly are the main objectives of smart energy technology. In this paper we would like to optimize the energy consumption of a residential building in Rome through proper python libraries and to predict future consumptions and temperatures using a tuned neural network model. In particular, we created our model thanks to the DesignBuilder software, while the optimization and energy simulation part was performed through BESOS, Energyplus and eppy. A set of sensors was emulated to retrieve data from the smart building. They were stored in an InfluxDB database and displayed by a user-friendly dashboard, Grafana. Once we had the simulated data, we were able to perform the energy signature and the prediction on future values of heating and cooling consumptions and temperatures. To accomplish this task, we operated a properly tuned neural network with LSTM neurons.
