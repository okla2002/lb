# lb
基于随机森林模型的锂离子电池剩余寿命预测

分工:
    
前期锂离子电池寿命预测方法调研：   
任务： 对锂离子电池寿命预测的现有方法进行深入调研，包括机器学习和传统物理建模方法的比较分析，了解它们的优缺点和适用场景。根据调研结果，基于数据特点和预测需求选择适合的模型   
成员：刘璇   
    
数据预处理和清洗、项目ppt制作:    
任务：负责从Excel文件中提取和处理锂电池数据，包括充电和放电阶段的数据提取，以及数据清洗和格式转换。    
代码：涉及使用 pandas 和 numpy 库处理数据，从Excel中读取数据并进行预处理，最终将处理后的数据保存为CSV文件。    
成员：辛星辰   
    
数据分析和可视化、项目报告撰写:   
任务：加载和可视化处理后的数据，绘制锂电池属性的变化图，如放电容量、内阻、充电时间等，帮助理解数据特征。并将预测结果与真实值进行可视化比较、可视化预测成果。   
代码：涉及使用 matplotlib 库进行数据可视化，绘制各属性随时间的变化图，例如放电容量的变化、内阻的变化等。   
成员：刘璇   
   
模型建立和训练：   
任务：负责建立机器学习模型，建立随机森林回归模型，预测锂电池的放电容量，并将预测结果与真实值进行比较。   
 代码：涉及使用 sklearn 库中的 RandomForestRegressor 进行模型建立和训练，包括数据的划分、模型的训练和评估，以及预测未来数据点的放电容量。    
 成员：包金赟   
   
代码优化调试:    
任务：计算测试集结果的误差、未来数据点预测、评估模型性能、在训练过程中调试模型并进行参数调优。   
代码：计算测试集预测结果的均方误差、调整随机森林模型的参数，如树的数量、树的深度等。   
成员：包金赟   
    
   
结果分析: 任务：负责分析模型预测结果的准确性和均方误差等评估指标，并对预测结果进行解释和讨论。    
 成员：刘璇，包金赟    
