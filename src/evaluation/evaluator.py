"""
Model evaluation and visualization tools
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Optional, Dict
import json
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations
    """

    def __init__(self, model: keras.Model, class_names: List[str],
                 output_dir: str = 'results'):
        """
        Initialize evaluator

        Args:
            model: Trained Keras model
            class_names: List of class names
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, test_generator, save_prefix: str = 'evaluation') -> Dict:
        """
        Comprehensive model evaluation

        Args:
            test_generator: Test data generator
            save_prefix: Prefix for saved files

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print("📊 Starting Model Evaluation")
        print(f"{'='*60}\n")

        # Get predictions
        print("🔮 Generating predictions...")
        y_pred = self.model.predict(test_generator, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Get true labels
        y_true = test_generator.classes

        # Calculate metrics
        print("\n📈 Calculating metrics...")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        self.plot_confusion_matrix(cm, save_prefix)

        # Classification Report
        report = classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        self.save_classification_report(report, save_prefix)

        # ROC Curves (for multi-class)
        self.plot_roc_curves(y_true, y_pred, save_prefix)

        # Precision-Recall Curves
        self.plot_precision_recall_curves(y_true, y_pred, save_prefix)

        # Per-class accuracy
        per_class_acc = self.calculate_per_class_accuracy(cm)
        self.plot_per_class_accuracy(per_class_acc, save_prefix)

        # Sample predictions visualization
        self.visualize_predictions(test_generator, save_prefix, num_samples=20)

        # Compile results
        results = {
            'overall_accuracy': float(report['accuracy']),
            'macro_avg_precision': float(report['macro avg']['precision']),
            'macro_avg_recall': float(report['macro avg']['recall']),
            'macro_avg_f1': float(report['macro avg']['f1-score']),
            'per_class_metrics': {
                class_name: {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1-score': float(report[class_name]['f1-score']),
                    'support': int(report[class_name]['support']),
                    'accuracy': float(per_class_acc[i])
                }
                for i, class_name in enumerate(self.class_names)
            }
        }

        # Save results
        results_path = os.path.join(
            self.output_dir, f'{save_prefix}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n✅ Evaluation complete! Results saved to: {self.output_dir}")
        print(f"\n📊 Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"📊 Macro Avg F1-Score: {results['macro_avg_f1']:.4f}")

        return results

    def plot_confusion_matrix(self, cm: np.ndarray, save_prefix: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Confusion matrix saved to: {save_path}")

    def save_classification_report(self, report: Dict, save_prefix: str):
        """Save classification report"""
        # Create text report
        report_text = "Classification Report\n"
        report_text += "=" * 60 + "\n\n"

        for class_name in self.class_names:
            metrics = report[class_name]
            report_text += f"{class_name}:\n"
            report_text += f"  Precision: {metrics['precision']:.4f}\n"
            report_text += f"  Recall: {metrics['recall']:.4f}\n"
            report_text += f"  F1-Score: {metrics['f1-score']:.4f}\n"
            report_text += f"  Support: {metrics['support']}\n\n"

        report_text += "\nOverall Metrics:\n"
        report_text += f"  Accuracy: {report['accuracy']:.4f}\n"
        report_text += f"  Macro Avg Precision: {report['macro avg']['precision']:.4f}\n"
        report_text += f"  Macro Avg Recall: {report['macro avg']['recall']:.4f}\n"
        report_text += f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n"

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_classification_report.txt')
        with open(save_path, 'w') as f:
            f.write(report_text)

        print(f"✅ Classification report saved to: {save_path}")

    def plot_roc_curves(self, y_true: np.ndarray, y_pred: np.ndarray, save_prefix: str):
        """Plot ROC curves for all classes"""
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ ROC curves saved to: {save_path}")

    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     save_prefix: str):
        """Plot precision-recall curves"""
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(10, 8))

        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred[:, i])
            plt.plot(recall, precision, lw=2, label=self.class_names[i])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_pr_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Precision-Recall curves saved to: {save_path}")

    def calculate_per_class_accuracy(self, cm: np.ndarray) -> List[float]:
        """Calculate per-class accuracy from confusion matrix"""
        per_class_acc = []
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum()
            else:
                acc = 0.0
            per_class_acc.append(acc)
        return per_class_acc

    def plot_per_class_accuracy(self, per_class_acc: List[float], save_prefix: str):
        """Plot per-class accuracy"""
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(self.num_classes), per_class_acc,
                       color='skyblue', edgecolor='navy')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(range(self.num_classes),
                   self.class_names, rotation=45, ha='right')
        plt.ylim([0, 1.0])
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{per_class_acc[i]:.3f}',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_per_class_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Per-class accuracy plot saved to: {save_path}")

    def visualize_predictions(self, test_generator, save_prefix: str,
                              num_samples: int = 20):
        """Visualize sample predictions"""
        # Get a batch of images
        test_generator.reset()
        batch_images, batch_labels = next(test_generator)

        # Select random samples
        num_samples = min(num_samples, len(batch_images))
        indices = np.random.choice(
            len(batch_images), num_samples, replace=False)

        # Make predictions
        predictions = self.model.predict(batch_images[indices])

        # Create figure
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(num_samples):
            ax = axes[i]

            # Display image
            img = batch_images[indices[i]]
            ax.imshow(img)

            # Get true and predicted labels
            true_label_idx = np.argmax(batch_labels[indices[i]])
            pred_label_idx = np.argmax(predictions[i])
            confidence = predictions[i][pred_label_idx]

            true_label = self.class_names[true_label_idx]
            pred_label = self.class_names[pred_label_idx]

            # Set title with color coding
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                         fontsize=8, color=color)
            ax.axis('off')

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(
            self.output_dir, f'{save_prefix}_sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Sample predictions saved to: {save_path}")


def plot_training_history(history: Dict, output_dir: str = 'results',
                          save_prefix: str = 'training'):
    """
    Plot training history

    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
        save_prefix: Prefix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss
    ax = axes[0, 0]
    ax.plot(history.get('loss', []), label='Train Loss', linewidth=2)
    ax.plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
    ax.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(history.get('accuracy', []), label='Train Accuracy', linewidth=2)
    ax.plot(history.get('val_accuracy', []), label='Val Accuracy', linewidth=2)
    ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)

    # Precision
    ax = axes[1, 0]
    if 'precision' in history:
        ax.plot(history.get('precision', []),
                label='Train Precision', linewidth=2)
        ax.plot(history.get('val_precision', []),
                label='Val Precision', linewidth=2)
        ax.set_title('Model Precision', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.axis('off')

    # Recall
    ax = axes[1, 1]
    if 'recall' in history:
        ax.plot(history.get('recall', []), label='Train Recall', linewidth=2)
        ax.plot(history.get('val_recall', []), label='Val Recall', linewidth=2)
        ax.set_title('Model Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.axis('off')

    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{save_prefix}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training history plot saved to: {save_path}")
