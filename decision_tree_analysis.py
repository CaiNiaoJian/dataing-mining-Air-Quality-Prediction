#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
决策树回归分析与可视化模块

该模块实现以下功能：
1. 决策树回归模型的构建与评估
2. 决策树结构的可视化
3. 特征重要性分析
4. 预测结果的多维度可视化
5. 残差分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义颜色方案
COLOR_PALETTE = ['#6e43b2', '#c9409f', '#fe5880', '#ff8a63', '#ffc256', '#f9f871']
sns.set_palette(COLOR_PALETTE)

class DecisionTreeAnalyzer:
    """决策树回归分析类"""
    
    def __init__(self, data_path='train.csv'):
        """
        初始化决策树分析器
        
        Args:
            data_path: 预处理后的数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        
        # 创建可视化结果目录
        if not os.path.exists('images/decision_tree'):
            os.makedirs('images/decision_tree')
    
    def prepare_data(self):
        """准备建模数据"""
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        
        # 选择特征
        feature_columns = [
            'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
            'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend',
            'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv'
        ]
        
        # 准备特征和目标变量
        self.X = self.data[feature_columns]
        self.y = self.data['Label']
        self.feature_names = feature_columns
        
        # 数据集划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print("数据准备完成：")
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
    
    def train_model(self, max_depth=5):
        """
        训练决策树模型
        
        Args:
            max_depth: 决策树最大深度
        """
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        
        # 预测和评估
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\n模型评估结果：")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        
        return y_pred
    
    def plot_tree_structure(self):
        """可视化决策树结构"""
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title("决策树结构可视化")
        plt.tight_layout()
        plt.savefig('images/decision_tree/tree_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存文本形式的决策树结构
        tree_text = export_text(self.model, feature_names=self.feature_names)
        with open('images/decision_tree/tree_structure.txt', 'w', encoding='utf-8') as f:
            f.write(tree_text)
    
    def plot_feature_importance(self):
        """可视化特征重要性"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('特征重要性排序（决策树）')
        bars = plt.bar(range(self.X_train.shape[1]), importances[indices])
        
        # 为每个柱子设置不同的颜色
        for i, bar in enumerate(bars):
            bar.set_color(COLOR_PALETTE[i % len(COLOR_PALETTE)])
            
        plt.xticks(range(self.X_train.shape[1]), 
                  [self.feature_names[i] for i in indices],
                  rotation=45)
        plt.tight_layout()
        plt.savefig('images/decision_tree/feature_importance.png')
        plt.close()
    
    def plot_learning_curve(self):
        """绘制学习曲线"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X, self.y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='训练集得分', color=COLOR_PALETTE[0])
        plt.fill_between(train_sizes, 
                        train_mean - train_std,
                        train_mean + train_std, 
                        alpha=0.1,
                        color=COLOR_PALETTE[0])
        
        plt.plot(train_sizes, test_mean, label='验证集得分', color=COLOR_PALETTE[1])
        plt.fill_between(train_sizes,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.1,
                        color=COLOR_PALETTE[1])
        
        plt.xlabel('训练样本数')
        plt.ylabel('R2得分')
        plt.title('决策树回归学习曲线')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/decision_tree/learning_curve.png')
        plt.close()
    
    def plot_prediction_analysis(self, y_pred):
        """
        预测结果分析可视化
        
        Args:
            y_pred: 预测结果
        """
        # 1. 预测值vs实际值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5, color=COLOR_PALETTE[0])
        plt.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 实际值')
        plt.tight_layout()
        plt.savefig('images/decision_tree/prediction_scatter.png')
        plt.close()
        
        # 2. 残差分析
        residuals = self.y_test - y_pred
        plt.figure(figsize=(12, 4))
        
        # 残差散点图
        plt.subplot(121)
        plt.scatter(y_pred, residuals, alpha=0.5, color=COLOR_PALETTE[1])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差散点图')
        
        # 残差分布图
        plt.subplot(122)
        sns.histplot(residuals, kde=True, color=COLOR_PALETTE[2])
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差分布图')
        
        plt.tight_layout()
        plt.savefig('images/decision_tree/residuals_analysis.png')
        plt.close()
        
        # 3. 预测值和实际值的分布对比
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.y_test, label='实际值', color=COLOR_PALETTE[3])
        sns.kdeplot(data=y_pred, label='预测值', color=COLOR_PALETTE[4])
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.title('预测值和实际值的分布对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig('images/decision_tree/distribution_comparison.png')
        plt.close()
    
    def analyze_depth_impact(self, max_depths=range(1, 11)):
        """
        分析树深度对模型性能的影响
        
        Args:
            max_depths: 要测试的树深度范围
        """
        train_scores = []
        test_scores = []
        
        for depth in max_depths:
            model = DecisionTreeRegressor(max_depth=depth, random_state=42)
            model.fit(self.X_train, self.y_train)
            
            train_score = r2_score(self.y_train, model.predict(self.X_train))
            test_score = r2_score(self.y_test, model.predict(self.X_test))
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(max_depths, train_scores, 'o-', label='训练集', color=COLOR_PALETTE[0])
        plt.plot(max_depths, test_scores, 'o-', label='测试集', color=COLOR_PALETTE[1])
        plt.xlabel('���的最大深度')
        plt.ylabel('R2得分')
        plt.title('树深度对模型性能的影响')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/decision_tree/depth_impact.png')
        plt.close()
    
    def run_analysis(self, max_depth=5):
        """
        执行完整的决策树分析流程
        
        Args:
            max_depth: 决策树最大深度
        """
        print("开始决策树回归分析...")
        
        # 1. 准备数据
        print("\n1. 准备数据...")
        self.prepare_data()
        
        # 2. 训练模型
        print("\n2. 训练模型...")
        y_pred = self.train_model(max_depth=max_depth)
        
        # 3. 可视化决策树结构
        print("\n3. 生成决策树结构可视化...")
        self.plot_tree_structure()
        
        # 4. 分析特征重要性
        print("\n4. 分析特征重要性...")
        self.plot_feature_importance()
        
        # 5. 绘制学习曲线
        print("\n5. 生成学习曲线...")
        self.plot_learning_curve()
        
        # 6. 预测结果分析
        print("\n6. 分析预测结果...")
        self.plot_prediction_analysis(y_pred)
        
        # 7. 分析树深度影响
        print("\n7. 分析树深度影响...")
        self.analyze_depth_impact()
        
        print("\n决策树回归分析完成！")

def main():
    """主函数"""
    print("="*50)
    print("开始决策树回归分析...")
    print("="*50)
    
    # 创建决策树分析器实例并执行分析
    analyzer = DecisionTreeAnalyzer(data_path='train.csv')
    analyzer.run_analysis(max_depth=5)

if __name__ == "__main__":
    main() 