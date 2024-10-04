# Machine Learning Workflow (Titanic Competition)  
# - A Template for Beginner

## Initial & Intermediary

### - Step 1
High Level Data Inspection for each column of the table(CSV, SqLite, BigQuery) to find following
- Missing Value
- Garbage Value

Age and Cabin contains null values

### - Step 2
Identify variables (colum/field) either  Continues or Classification

**Continues Variables**
- PassengerID
- Name
- Age
- Ticket
- Fare
- Cabin

**Classification Variables**
- Survived
- Pclass
- Sex
- SibSp
- Parch
- Embarked

### - Step 3

**Data Insight & Analytics**

Learn From Domain

**Pclass:** A proxy for socio-economic status (SES)
- 1st = Upper
- 2nd = Middle
- 3rd = Lower

**Age:** Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**Sibsp:** The dataset defines family relations in this way...
- Sibling = brother, sister, stepbrother, stepsister
- Spouse = husband, wife (mistresses and fiancés were ignored)

**Parch:** The dataset defines family relations in this way...
- Parent = mother, father
- Child = daughter, son, stepdaughter, stepson
- Some children travelled only with a nanny, therefore parch=0 for them.

Using Tools
- Pandas
- matplotlib
- SeaBorn

**Note:** BigQuery can also be used at this step. 

# Feature Engineering
