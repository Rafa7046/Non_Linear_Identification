# Non Linear Identification

# Data description

The data used to test the algorithms were gathered online from the [DaISy](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)
and [Nonlinear Benchmarks](https://www.nonlinearbenchmark.org/benchmarks) databases. In the following we give a brief description of each
dataset.

## Ball and beam

Data in the `ball-and-beam.csv` file of a the ball and beam practicum at ESAT-SISTA.

* Sampling time := 0.1 sec
* Number of samples := 1000 samples
* Inputs
    * u := angle of the beam
* Outputs
    * y := position of the ball
* Columns
    1. input u
    2. output y

## Liquid-saturated steam heat exchanger 

Data in the `exchanger.csv` file of a liquid-satured steam heat exchanger, where water is
heated by pressurized saturated steam through a copper tube. The output variable is the
outlet liquid temperature. The input variables are the liquid flow rate, the steam
temperature, and the inlet liquid temperature.

* Sampling time := 1 s
* Number of samples := 4000 samples
* Inputs
    * q := liquid flow rate 
* Outputs
    * th := outlet liquid temperature
* Columns
    1. time-steps ts
    2. input q
    3. output th 

## Data from a flexible robot arm

Data in the `robot-arm.csv` file from a flexible robot arm. The arm is installed on an
electrical motor. The transfer function has been modeled from the measured reaction 
torque of the structure on the ground to the acceleration of the flexible arm. The applied
input is a periodic sine sweep.

* Sampling time := _unknown_
* Number of samples := 1024 samples
* Inputs
    * u := reaction torque of the structure
* Outputs
    * y := accelaration of the flexible arm
* Columns
	1. input u
	2. output y

## Data from Cascaded Tanks with Overflow

Data in `tanque.csv` file from Cascaded Tanks with Overflow. The cascaded tanks system is a fluid level control system consisting of two tanks with free outlets fed by a pump. The input signal controls a water pump that delivers the water from a reservoir into the upper water tank. The water of the upper tank flows through a small opening into the lower tank, and finally through a small opening from the lower tank back into the reservoir. 

* Sampling time := 4 s
* Number of samples := 1024 samples
* Inputs
    * uVal := Tension
* Outputs
    * yVal := Level of water
* Columns
	1. uEst
	2. uVal
 	3. yEst
 	4. yVal
  	5. Ts

 ## Data from Silverbox System

Data in `SNLS80mV.csv` file from Silverbox System. The Silverbox system can be seen as an electronic implementation of the Duffing oscillator. It is build as a 2nd order linear time-invariant system with a 3rd degree polynomial static nonlinearity around it in feedback. This type of dynamics are, for instance, often encountered in mechanical systems.

* Sampling time := _unknown_
* Number of samples := 131073 samples
* Inputs
    * V1 := is the input record 
* Outputs
    * V2 := is the measured output
* Columns
    1. V1
    2. V2
