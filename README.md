#  基于OCR的身份证要素提取

        此文档用于复现2019BDCI赛道“基于OCR的身份证要素提取”的比赛成绩。之后我们会将其开源在github上.
        我们利用基本的图像识别技术，利用基于CNN的pytorch搭建了整套系统，对身份证上要素的识别有较高的准确率。
        
   
   

## 要求


```bash
pip install -r requirements.txt
```

## 数据预处理
```bash
cd preprocess
```
在preprocess文件夹中，各个py文件的作用如下:       
| py文件 | 作用 |        
| -------- | -------- |                    
| dealPSERes.py | 处理PSENet检测完身份证后的结果，获得标准矩形大小的身份证图片 |                           
|make_single_word.py|将身份证背面的地址和正面的签发机关拆分为标准模板框单字，生成对应的标注csv文件|       
 |make_another_word.py|将身份证背面的姓名、性别、民族拆分为标准模板框单字，生成对应的标注csv文件|      
|make_single_number.py|将身份证上的id信息拆分为单数字，拆分年、月、日为标准模板框单字，生成对应的标注csv文件|     
|csv_merge.py|将单字的标注csv文件融合。并将初赛的csv文件与复赛的csv文件一一对应融合|          
|train_val_split.py|将数据集分为训练集和验证集|           
|add_train.py|在单字的训练集上加上常见字|                
|add_year_train.py|在年的训练集上加上所有的年份图片，以防止训练集中有些年份没有出现|               

--------------------

* 初始得到的身份证不是标准形状的身份证，我们需要将原始数据中的身份证框检测出来，这里使用PSNET模型，将身份证正反两面检测出来。              



* 分割单字
图片校正完后，需要对标准矩形的身份证截取其对应的模版框:    

1、截取地址和签发机关单字模板框，执行：    
    
```bash
python make_single_word.py
```
生成**address_authority.csv**文件    

-----------------
2、将身份证背面的姓名、性别、民族拆分为单字模板框，执行：  
```bash
python make_another_word.py
```   
生成**total_other_word.csv**文件              

-------------------
3、切割数字模板框，分别制作id、年、月、日的模板框，执行：   
```bash
python make_single_number.py
```
生成id_number.csv, number_year.csv, number_month.csv, number_day.csv          

--------------------
4、确保文件名无误后，同时确保当前目录中存在初赛的相应数据集csv文件(s1_address_authority.csv、s1_total_other_word.csv、S1_id_number.csv、S1_number_year.csv、S1_number_month.csv、S1_number_day.csv）,分别将其合并，执行：    
```bash
python csv_merge.py
```
生成total_word.csv、year_total.csv、month_total.csv、day_total.csv等**总数据集**文件          

------------------
5、划分训练集和验证集，需要分别将total_word.csv、id_total.csv,year_total.csv、month_total.csv、day_total.csv四个文件划分为5个训练集加5个验证集，依次修改train_val_split.py中的csv_path_prefix参数，分别将其修改为total_word、id_total、year_total、month_total、day_total,执行**5次**脚本：
```bash
python train_val_split.py
```
得到单字数据集（total_word_train.csv、total_word_val.csv）、id(身份证)数据集(id_total_train.csv、id_total_val.csv)、年数据集（year_total_train.csv、year_total_val.csv)、月数据集（month_total_train.csv、month_total_val.csv)、日数据集（day_total_train.csv、day_total_val.csv）    
**这5份数据集用作训练5个模型来预测身份证要素**             

-------------------
6、增加单字训练集：    
由于训练集中可能存在一些没有出现过的单字，导致最后识别错误，我们利用身份证背景，生成一些常见字，加入到训练集中，执行：  
```bash
python add_train.py
```
生成最终的单字训练集：**total_word_final_train.csv**                

---------------
7、增加年数据集
我们预测的年分类是从1900-2050年，故又可能训练集中不存在某一些年份，所以我们利用身份证模板给每个年份增加了一些样本，执行：  
```bash
python add_year_train.py
```       
生成最终的年训练文件：**year_total_final_train.csv**             

------------------
8、最终的模型和对应的训练集验证集 
在这个识别问题上，我们都是用resnet50作为backbone              
| 模型 |  训练集csv文件 | 验证集csv文件 |
| ---- | ---- | ---- |
|单字模型 | total_word_final_train.csv|total_word_val.csv|
|id模型|id_total_train.csv|id_total_val.csv|
|年模型| year_total_final_train.csv| year_total_val.csv|
|月模型| month_total_train.csv| month_total_val.csv|
|日模型| day_total_train.csv| day_total_val.csv|

    
## 训练及预测
回到项目根目录                
```bash
cd train
```
在train文件夹中，各个文件的含义如下          
|文件|作用或含义|
|----|----|
|alphabet_word.txt|单字分类的类别字典|
|resnet_word_train.py|单字模型训练文件，用于训练单字|
|resnet_number_train.py|数字模型训练文件，用于训练id、年、月、日|
|resnet_dataset.py|resnet的数据集文件，重定义了如何从csv文件中获取sample|
|myResnet.py|resnet模型文件，可以从该文件中获取模型|
|default.py|可以调整的超参数，复现的话只需要修改batchsize|
|utils.py|一些工具|
|merge_predict.py|预测文件，可以预测姓名、性别、民族、地址、签发机关字段|
|single_number_predict.py|预测数字，可以预测年、月、日、id、有效期|  

### 训练
可以将预处理中的训练集和验证集csv文件放入train目录中，  
执行resnet_word_train.py和resnet_number_train.py，期中各个参数的意思是：    
    1、--trainPath: 训练集csv文件路径
    2、--valPath: 验证集csv文件路径
    3、--resume:指定从某个已经训练好的模型之后开始训练
    4、--checkpoint:指定模型存放的目录  
其中，resnet_number_train.py可以通过--trainFlag来指定训练哪个模型，其中0表示训练年、1表示训练月、2表示训练日、3表示训练id。

---------
* 单字训练：
```bash
python resnet_word_train.py --trainPath total_word_final_train.csv --valPath total_word_val.csv --checkpoint checkpoint/resnet_word
```
最终的模型将会保存在checkpoint目录下的resnet_word中
* 数字训练       
1、年训练：
```bash
python resnet_number_train.py --trainPath year_total_final_train.csv --valPath year_total_val.csv --trainFlag 0 --checkpoint checkpoint/resnet_year
```
最终的模型将会保存在checkpoint目录下的resnet_year中
2、月训练：
```bash
python resnet_number_train.py --trainPath month_total_train.csv --valPath month_total_val.csv --trainFlag 1 --checkpoint checkpoint/resnet_month
```
最终的模型将会保存在checkpoint目录下的resnet_month中
3、日训练：
```bash
python resnet_number_train.py --trainPath day_total_train.csv --valPath day_total_val.csv --trainFlag 2 --checkpoint checkpoint/resnet_day
```
最终的模型将会保存在checkpoint目录下的resnet_day中  
4、id训练
```bash
python resnet_number_train.py --trainPath id_total_train.csv --valPath id_total_val.csv --trainFlag 3 --checkpoint checkpoint/resnet_id
```
最终的模型将会保存在checkpoint目录下的resnet_id中

### 预测
1、预测单字  
将准确率最高的模型单字名称填入merge_predict.py中的Resnet_model_path中，执行：   
```bash
python merge_predict.py
```
生成**word_resnet_result.csv**
2、预测数字  
填写single_number_predict.py中的id模型、年模型、月模型、日模型的路径，执行:
```bash
python single_number_predict.py
```
生成**total_number_resnet_result.csv**

## 后处理（规则矫正）
回到根目录；  
```bash
cd final_treatment
```
final_treatment文件夹内各个文件的含义：
|文件|作用或含义|
|----|----|
|nation_filter.py|民族校正，将预测后不在56个民族内的字段进行编辑距离校正|
|gender_filter.py|将性别预测不为男女的字段重新赋值为男或女|
|res_merge.py|将之前预测的单字结果和数字结果融合|
|correct_result.py|做签发机关校正、地址校正、做身份证号（与年月日）的适配|
|correct_period.py|做有效期的校正|

我们目前得到的结果文件是：word_resnet_result.csv(姓名、民族、性别、地址、签发机关）和 total_number_resnet_result.csv（年、月、日、id、有效期）
1、民族校正
执行：  
```bash
python nation_filter.py
```
2、合并结果
执行：  
```bash
python res_merge.py
```
得到结果文件：**final_res.csv**  
3、性别校正
执行：
```bash
python gender_filter.py
```
得到结果文件：**final_res_gender.csv**  
4、地址校正、签发机关校正、id与年月日的校正
* 地址校正和签发机关校正的原理我们将我们预测的地址数据放入官方的地址和签发机关进行一个距离相似度匹配，然后对地址进行校正。  
* 身份证id与年月日的匹配我们利用的是水印的位置，当发生身份证id上的数字对应的年月日和预测的实际年月日不符时，我们利用水印的位置用一方去校正另外一方。  
执行：
```bash
python correct_result.py
```
得到结果文件：**final_res_date.csv**  
5、有效期校正  
* 有效期匹配的原理是身份证的有效期有一定的规则，当小于16岁时为5年，大于16岁小于26岁时为10年，26岁到45岁为20年，46岁及以上为长期,同时若不是长期有效，则前后对应的月和日应该相等，我们根据水印的位置判断，当有效期间隔不对或月日不匹配时，利用水印的位置去校正有效期，执行：
```bash
python correct_period.py
```
得到最终提交的结果文件：**final_res_final.csv**

