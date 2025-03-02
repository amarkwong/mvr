# How to use this

## Data requirements

- Header should follow the naming pattern as `column name #description`, where the description can be omited if the data is self explanatory. Otherwise, the description could be the posible value of this column

- Some not proper formatted data will be fine, such as using percentage and number in same row. 

- Ideally all the data points should not be empty, but we have implemented some `smart` autofilling, which allows user to choose from 

    - Drop, drop the data row with the data point missing
    - Mean, use the average value to mock the data
    - Median, use the median value to mock the data
    - Mode, use the mode value to mock the data
    - Zero, use the 0 value to mock the data

    it can be set in conf/data_fitting.json
