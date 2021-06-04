from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

for cellline_i in cellline_list:
  for epigenomic_i in epigenomic_list:
    encode = webdriver.Chrome(path_to_chrome + 'chromedriver.exe')
    encode.get('https://www.encodeproject.org/search/?type=Experiment&control_type!=*&status=released&perturbed=false')
    encode.find_element_by_xpath("//span[.='Home sapiens']").click()
    encode.find_element_by_xpath("//span[.=" + cellline_i + "]").click()
    encode.find_element_by_xpath("//span[.=" + cellline_i + "]").click()
  


