"""
Production deployment script
"""

import os
import sys
import argparse
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlops.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description='Deploy model to production')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--version', type=str, default=None,
                       help='Model version (latest if not specified)')
    parser.add_argument('--source-stage', type=str, default='staging',
                       choices=['development', 'staging'],
                       help='Source stage')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ML GEOMETRY - PRODUCTION DEPLOYMENT")
    print("=" * 80)
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Get model
    print(f"\n🔍 Looking for model: {args.model_name}")
    model_entry = registry.get_model(
        model_name=args.model_name,
        version=args.version,
        stage=args.source_stage
    )
    
    if not model_entry:
        print(f"❌ Model not found in {args.source_stage} stage")
        return
    
    print(f"\n✅ Found model:")
    print(f"   Registry ID: {model_entry['registry_id']}")
    print(f"   Version: {model_entry['version']}")
    print(f"   Current Stage: {model_entry['stage']}")
    print(f"   Path: {model_entry['path']}")
    
    if 'metadata' in model_entry:
        print(f"\n📊 Metrics:")
        for key, value in model_entry['metadata'].items():
            print(f"   {key}: {value}")
    
    # Deployment checklist
    print("\n📋 Pre-deployment Checklist:")
    
    checklist = {
        'Model exists': os.path.exists(model_entry['path']),
        'Model in staging': model_entry['stage'] == 'staging',
        'Has accuracy metric': 'accuracy' in model_entry.get('metadata', {}),
        'Accuracy > 90%': model_entry.get('metadata', {}).get('accuracy', 0) > 0.9
    }
    
    for check, passed in checklist.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    all_passed = all(checklist.values())
    
    if not all_passed:
        print("\n⚠️  Warning: Not all checks passed!")
        if not args.dry_run:
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Deployment cancelled")
                return
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        print(f"\nWould promote: {model_entry['registry_id']}")
        print(f"From: {model_entry['stage']}")
        print(f"To: production")
        return
    
    # Promote to production
    print(f"\n🚀 Promoting to production...")
    success = registry.promote_model(
        registry_id=model_entry['registry_id'],
        target_stage='production'
    )
    
    if success:
        print("\n" + "=" * 80)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("=" * 80)
        print(f"\nModel {model_entry['registry_id']} is now in PRODUCTION")
        print("\n📝 Next steps:")
        print("   1. Update API server to use new model")
        print("   2. Run smoke tests")
        print("   3. Monitor performance metrics")
        print("   4. Set up alerts")
    else:
        print("\n❌ Deployment failed!")


if __name__ == '__main__':
    main()
