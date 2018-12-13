1. Our program was made using opencv, numpy, scipy, matplotlib libraries. Our python  version was 3.6.7 .The links to install each of these libraries are listed below:

opencv: https://docs.opencv.org/3.4.3/df/d65/tutorial_table_of_content_introduction.html
numpy and scipy:   http://www.numpy.org/
matplotlib: https://matplotlib.org/users/installing.html
python 3.6: https://www.python.org/downloads/

2.How to run Our codes:

--In Windows(I used Windows 10):
  
  step1: open 'cmd' or 'powershell' terminal, 
  type `activate your_environment_name` to activate environment. (if you don't have an environment , just ignore this line.)
  
  step2: go to the folder where you stored our code, type `python part1-main.py or python part-2-buildin-function.py or python part2-own-produced.py` (if you have two versions of python, type `python3 main.py`, maybe) to run the file. You must add `python.exe` to your PATH environmental variable, otherwise you have to use the full path of the python interpreter. All our programs are written in this file.

-- In Mac and Linux:
   
   step1: open 'Terminal', then go to the folder where you have stored our python script.
   step2: type ``python part1-main.py or python part-2-buildin-function.py or python part2-own-produced.py` to run the program.

3. What about the own-produced programme:
  
    We implemented our programme with the ¡®get_harris_position¡¯ in the first part, after getting the interest point in both pictures, we took patches on each interest points with ¡®get_patch¡¯ and ¡®extract_patch¡¯ functions, after this, we calculated the SSD numbers, and stored them in a list called ¡°similarity¡±, after sorting them, we took the smallest and the second smallest SSDs to compare, if they satisfy the condition that we mentioned before, we choose them as the accepted matches. We also implemented the reversed comparison, taking the patches in image 2 as the beginning patches to match, we got the reversed matching. After this, we created a eliminate-multi-matches funtion to try to eliminate the multi-matches. 



