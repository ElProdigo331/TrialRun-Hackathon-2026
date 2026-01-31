# Energy A.I. Hackathon 2026, the 6th Annual Event

## Hosts: [Prof. Michael Pyrcz](https://x.com/GeostatsGuy) and [Prof. John T. Foster](https://x.com/johntfoster)

### Architects: [Dinghan Wang](https://www.linkedin.com/in/dinghan-wang-b1669b32b/) and [Nataly Chacon-Buitrago](https://www.linkedin.com/in/nataly-chacon-buitrago/)

### Home Department Chair and Hackathon Supporter: [Prof. Matthew Balhoff](https://www.linkedin.com/in/matthew-balhoff-4297b247/)


### Sponsors 

Platinum:
[Phillips66](https://www.phillips66.com/), [Chevron](https://www.chevron.com), [ConocoPhillips](https://www.conocophillips.com/), [Oxy](https://www.oxy.com/about/)

Gold:
[ExxonMobil](https://corporate.exxonmobil.com/), [Aramco Americas](https://americas.aramco.com/en)

Bronze:
[Elk Range Roylaties](https://www.elkrange.com/), [ComboCurve](https://combocurve.com/)

### Coordination and Student Engagement: Rowan Halliday, Gabby Banales, and Stacia Miller

### Appreciation to the Hildebrand Department of Petroleum and Geosystems Engineering for their great support of the Energy AI Hackathon 

___

### Energy A.I. Hackathon 2026 Problem Description 

**Goal**: Develop a data analytics and machine learning workflow in Python to:

* Predict **Cumulative three years oil production** for preproduction wells. This includes single estimates and an uncertainty model represented by 100 realizations for each well. All preproduction wells start producing on December 1st, 2022 (so we can score your predictions and uncertainty model against the production from December 1st, 2022 to December 1st, 2025)
 
#### Background

Accurate forecasting of oil production is a fundamental challenge in subsurface energy development and reservoir management. In many fields, operators rely on vertical wells drilled across a large reservoir to characterize geological variability and assess production potential. While detailed well-log measurements are routinely collected during drilling, production data are only available for wells that have already been placed on production.

In practice, production wells often begin producing at different times and have uneven production histories, resulting in datasets where wells contain varying lengths of monthly production records. Over time, continued production can influence the performance of nearby wells. New wells drilled within the same field are referred as infill wells.

We challenge the Energy A.I. Hackathon 2026 teams of The University of Texas at Austin to build a data analytics and machine learning workflow to predict cumulative three years oil production with uncertainty for preproduction new wells.

This will require:

* the understanding of geological and enginnering features by using petroleum domain knowledge and integration of domain expertise at every step
* feature engineering including feature selection, feature projection, feature transformations, and feature imputation to address missing data
* selection, training and tuning of robust machine learning prediction models to calculate estimates with an uncertainty model
* checking your models and visusalizing and interpreting your model results
___

### Available Data Files Inventory

You have the following available data:

* **Well Logs**
  These two files contain the well log data along the wellbore for all 83 wells.

  Well_log_data_production_wells.csv - well logs for the previous production wells, well indices from 1 to 71, production is known for these wells
  Well_log_data_preproduction_wells.csv - well logs for the remaining, preproduction wells, well indices from 72 to 83, production is NOT known and you will prediction 3 year cumulative production with uncertainty for these wells.

* **Well Log data explanation**
  This file contains the data explanation of all columns in two well log files.

* **Map Data**
  The following map data is available:

  2d_sand_propotion.npy - proportion of sand over the vertical column, 2D facies proportion map. This map was calculated by inversion of available seismic data. Note, the resolution is not high and there is noise.

* **Production History**
  The following production history is available:

  Production_history_production_wells.csv - the monthly cumulative oil, water and gas productions for the 71 previous production wells with different production starting time.
___

### Required Hackathon Submissions

By January 25 at 12:00 noon each team must submit:

* **Solution Table** - a .csv file with your predictions for the 12 preproduction wells, estimates and uncertainty model realizations. The submitted file should follow the format of the provided template solution.csv for automatic scoring. Use [solution.csv](solution.csv) file (in this directory) to record your solution.

    * the file must be named `solution.csv` with final values in a commit and then pushed to Github for automated scoring.


* **Python Workflow and Associated Files** - committed to this repository with the workflow as a Jupyter Notebook `.ipynb` file along with all data files required to reproduce your team's solutions. The submitted workflow Jupyter Notebook should follow the format of the provided template [Hackathon_Project_Template](Hackathon_Project_Template.ipynb) for enhanced workflow communication and code readibility.

    * the file must be named `[Team_Name].ipynb` and pushed to GitHub for review and scoring (for code readability) by the hackathon architects. For example, if you team name is 'Longhorns', then this file must be Longhorns.ipynb.


* **Presentation** - a PowerPoint slide deck .pptx file for your team's final presentation to our judges. The submitted presentation should follow the format of the provided example presentation [Hackathon_Presentation_Template](Hackathon_Presentation_Template.pptx).

    * the file must be named '[Team_Name].pptx` and pushed to GitHub for review by the judges. For example, if your team name is 'Longhorns', then this file must be Longhorns.pptx
 
    * must be a .pptx file, do not convert to .pdf 

    * **YOU MUST INCLUDE ALL TEAM PARTICIPANTS NAMES ON YOUR PRESENTATION TITLE SLIDE**


