Initial work on the project. Some key things to note:
- last year and earlier, the event rankings website used a Captcha "i'm not a robot" system for accessing biogs so my initial build scraped each year that a swimmer was active to find their times from the ALL-TIME rankings, which is slow and laborious.
-    As the Captcha system is no longer in use, I developed a more robust, less buggy version, hence its title of Optimized Scraper
- the plotter just uses fixed values but can be adjusted to select different files to plot by changing file names and the labels in the relevant functions
- the scraper makes use of functional conventions to allow a modular and easy-to-debug program flow which is helpful for in future if the rankings site changes slightly,
-   such as bringing back the Captcha system, as then specific functions only need to be modified to run and as most of the querying and scraping relies on currying functions, it should be straightforward to adjust the code to suit site changes.
- some sample plots using the plotter by changing file names and labelling are included in the Plots folder
