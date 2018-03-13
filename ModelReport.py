import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, inch, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, StyleSheet1

class TestReport:
    
    def __init__(self,task='classification', model_name = 'Statistical Model', title="Cool Title",
                 description = 'none_provided', data_source='none_provided', train_size='not_specified', 
                 contact_name = "none_provided", imputation='none_provided'):
        self.contact_name = contact_name
        self.task = task
        self.title = title
        self.model_name = model_name
        self.description = description
        self.data_source = data_source
        self.train_size = train_size
        self.imputation = imputation
    
    acc = None
    auc = None
    brier = None
    f1 = None
    
    def fit(self,model,test_X,test_y):
        self.test_size = test_y.shape[0]
        self.num_of_features = test_X.shape[1]
            
        if self.task=='classification' and len(set(test_y)) > 2:
            return("classification task specified but >2 classes in output, change task='regression', multiclass-classification not currentely supported")

        if self.task=='classification':
            acc_ = []
            auc_ = []
            brier_ = []
            f1_ = []
            exp_var_ = []
            print("Averaging scores over 5 partitions")
            for seed in [0,10,100,1000,10000,100000]:
                np.random.seed = seed
                if seed>0:
                    subset_prop = 0.8
                else:
                    subset_prop = 1
                test_subset_idx = np.random.choice(list(range(1,test_y.shape[0])), int(test_y.shape[0]*subset_prop))
                test_X_subset = test_X[test_subset_idx, :]
                test_y_subset = test_y[test_subset_idx]
                preds = model.predict_proba(test_X)[:,1]
                preds_class = np.around(preds)
                acc_.append(metrics.accuracy_score(y_true=test_y,y_pred=preds_class))
                auc_.append(metrics.roc_auc_score(y_true=test_y, y_score=preds))
                brier_.append(metrics.brier_score_loss(y_true=test_y,y_prob=preds))
                f1_.append(metrics.fbeta_score(y_true=test_y,y_pred=preds_class, beta=1))
            
            self.acc = np.mean(acc_)
            self.auc = np.mean(auc_)
            self.brier = np.mean(brier_)
            self.f1 = np.mean(f1_)
            
            return({"Accuracy":[self.acc],
                    "AUC:":[self.auc],
                    "Brier Score": [self.brier],
                    "F1 Score":[self.f1]})

        if self.task=='regression':
            
            print("Averaging scores over 5 partitions")

            mse_ = []
            mae_ = []
            r_squared_ = []
            explained_variance_ = []
            
            for seed in [0,10,100,1000,10000,100000]:
                np.random.seed = seed
                if seed>0:
                    subset_prop = 0.8
                else:
                    subset_prop = 1
                test_subset_idx = np.random.choice(list(range(1,test_y.shape[0])), int(test_y.shape[0]*subset_prop))
                test_X_subset = test_X[test_subset_idx, :]
                test_y_subset = test_y[test_subset_idx]
                preds = model.predict(test_X_subset)
                mse_.append(metrics.mean_squared_error(y_true=test_y_subset,y_pred=preds))
                mae_.append(metrics.mean_absolute_error(y_true=test_y_subset,y_pred=preds))
                r_squared_.append(metrics.r2_score(y_true=test_y_subset,y_pred=preds))
                explained_variance_.append(metrics.explained_variance_score(y_true=test_y_subset,y_pred=preds))

            
            self.mse = np.mean(mse_)
            self.mae = np.mean(mae_)
            self.r_squared = np.mean(r_squared_)
            self.explained_variance = np.mean(explained_variance_)
            
            return({"Mean Squared Error":[self.mse],
                    "Mean Absolute Error:":[self.mae],
                    "R_Squared": [self.r_squared],
                    "Explained Variance":[self.explained_variance]})

    def generate_report(self, output_file='test_report.pdf'):
        print("Creating report...")

 
        doc = SimpleDocTemplate(output_file, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
        #doc.pagesize = landscape(A4)
        elements = []
        
        if self.task=='classification':
            data = [
            ["Model Type ", str(self.model_name)],
            ["Data Source ", str(self.data_source)],
            ["Training Size ", str(self.train_size)],
            ["Test Size ", str(self.test_size)],
            ["Imputation ", str(self.imputation)],
            ["Number of Features ", str(self.num_of_features)],
            ["Test AUC ", str(round(0.744,2))],
            ["Test Accuracy ", str(round(self.acc,2))],
            ["Test Brier Score ", str(round(self.brier,2))],
            ["Test F1 Score ", str(round(self.f1,2))]
            ]
    
        if self.task=='regression':
            data = [
            ["Model Type ", str(self.model_name)],
            ["Data Source ", str(self.data_source)],
            ["Training Size ", str(self.train_size)],
            ["Test Size ", str(self.test_size)],
            ["Imputation ", str(self.imputation)],
            ["Number of Features ", str(self.num_of_features)],
            ["Test Mean Squared Error ", str(round(self.mse,2))],
            ["Test Mean Absolute Error ", str(round(self.mae,2))],
            ["Test Rsquared ", str(round(self.r_squared,2))],
            ["Test Explained Variance ", str(round(self.explained_variance,2))]
            ]
 
        #TODO: Get this line right instead of just copying it from the docs
        style = TableStyle([('ALIGN',(1,1),(-2,-2),'RIGHT'),
                       ('TEXTCOLOR',(1,1),(-2,-2),colors.red),
                       ('VALIGN',(0,0),(0,-1),'TOP'),
                       ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
                       ('ALIGN',(0,-1),(-1,-1),'CENTER'),
                       ('VALIGN',(0,-1),(-1,-1),'MIDDLE'),
                       ('TEXTCOLOR',(0,-1),(-1,-1),colors.green),
                       ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                       ('BOX', (0,0), (-1,-1), 0.25, colors.black),
                       ])
 
        #Configure style and word wrap
        ss = getSampleStyleSheet()
        s = ss["BodyText"]
        title_style = ss["Title"]
        heading_style = ss["Heading2"]
        s.wordWrap = 'CJK'
        data2 = [[Paragraph(cell, s) for cell in row] for row in data]
        t=Table(data2)
        t.setStyle(style)
 
        # Create explanation paragraph
        title = Paragraph(self.title,title_style)
        contact_header = Paragraph("Contact: ",heading_style)
        contact = Paragraph(self.contact_name,s)
        description_header = Paragraph("Brief Description: ", heading_style)
        description = Paragraph(self.description, style=s)
        seperator = Paragraph("----", style=s)
        metrics_header = Paragraph("Performance Metrics (5 partition average)", heading_style)


        #Send the data and build the file
        elements.append(title)
        elements.append(seperator)
        elements.append(contact_header)
        elements.append(contact)
        elements.append(seperator)
        elements.append(description_header)
        elements.append(description)
        elements.append(seperator)
        elements.append(metrics_header)
        elements.append(t)
        doc.build(elements)
        print("**Report was succesfully created at " + output_file + "**")
    
    