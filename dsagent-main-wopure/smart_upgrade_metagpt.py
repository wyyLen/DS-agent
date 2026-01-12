"""
智能升级MetaGPT到最新版本
保留DSAgent特有的模块和修改
"""

import os
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent
old_metagpt = project_root / "metagpt"
new_metagpt = project_root / "MetaGPT-latest" / "metagpt"
backup_dir = project_root / "metagpt_backup"

# DSAgent特有的文件和目录（需要保留）
DSAGENT_SPECIFIC = [
    "roles/ds_agent/",
    "actions/ds_agent/",
    "rag/engines/customSolutionSamplesGenerate.py",
    "rag/engines/customMixture.py",
    "rag/engines/customWorkflowGM.py",
    "rag/engines/customEmbeddingComparisonEngine.py",
    "rag/engines/graphUtils.py",
    "rag/engines/GraphMatching/",
    "strategy/ds_planner.py",
    "strategy/ds_task_type.py",
    "strategy/lats_react.py",
    "strategy/lats_react_stream.py",
    "prompts/ds_agent/",
]

# 需要合并的文件（既有新版本又有DSAgent修改）
FILES_TO_MERGE = [
    "const.py",
]

print("=" * 80)
print("MetaGPT 智能升级工具")
print("=" * 80)
print()

# 步骤1: 备份现有metagpt
print("步骤 1: 备份现有 metagpt 目录...")
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(old_metagpt, backup_dir)
print(f"✓ 已备份到: {backup_dir}")
print()

# 步骤2: 删除旧的metagpt（保留.gitignore等）
print("步骤 2: 清理旧 metagpt 目录...")
for item in old_metagpt.iterdir():
    if item.name not in ['.git', '.gitignore']:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
print("✓ 清理完成")
print()

# 步骤3: 复制新版MetaGPT
print("步骤 3: 复制最新 MetaGPT...")
for item in new_metagpt.iterdir():
    dest = old_metagpt / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)
print("✓ 最新版本已复制")
print()

# 步骤4: 恢复DSAgent特有文件
print("步骤 4: 恢复 DSAgent 特有文件...")
restored_count = 0
for specific_path in DSAGENT_SPECIFIC:
    source = backup_dir / specific_path
    target = old_metagpt / specific_path
    
    if not source.exists():
        print(f"  ⚠ 未找到: {specific_path}")
        continue
    
    # 创建目标目录
    target.parent.mkdir(parents=True, exist_ok=True)
    
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        print(f"  ✓ 目录: {specific_path}")
    else:
        shutil.copy2(source, target)
        print(f"  ✓ 文件: {specific_path}")
    
    restored_count += 1

print(f"✓ 已恢复 {restored_count} 个特有文件/目录")
print()

# 步骤5: 处理需要合并的文件
print("步骤 5: 合并关键配置文件...")
for file_to_merge in FILES_TO_MERGE:
    old_file = backup_dir / file_to_merge
    new_file = old_metagpt / file_to_merge
    
    if not old_file.exists():
        continue
    
    print(f"  处理: {file_to_merge}")
    
    # 读取旧文件中的DSAgent特有常量
    with open(old_file, 'r', encoding='utf-8') as f:
        old_content = f.read()
    
    # 读取新文件
    with open(new_file, 'r', encoding='utf-8') as f:
        new_content = f.read()
    
    # 提取并添加DSAgent特有的常量定义
    dsagent_additions = []
    
    if 'AGENT_SERVICE_FILE' in old_content and 'AGENT_SERVICE_FILE' not in new_content:
        dsagent_additions.append('\n# DSAgent specific paths')
        dsagent_additions.append('AGENT_SERVICE_FILE = EXAMPLE_PATH / "ds_agent" / "agent_service" / "uploads"')
    
    if 'EXP_PLAN' in old_content and 'EXP_PLAN' not in new_content:
        dsagent_additions.append('EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank" / "plan_exp.json"')
    
    if 'WORKFLOW_EXP' in old_content and 'WORKFLOW_EXP' not in new_content:
        dsagent_additions.append('WORKFLOW_EXP = EXAMPLE_DATA_PATH / "exp_bank" / "workflow_exp2_clean.json"')
    
    if 'CUSTOM_DA_EVAL_RES_FILE' in old_content and 'CUSTOM_DA_EVAL_RES_FILE' not in new_content:
        dsagent_additions.append('CUSTOM_DA_EVAL_RES_FILE = EXAMPLE_DATA_PATH / "exp_bank" / "custom_da_eval_res.json"')
    
    if 'DI_EVAL_RES_FILE' in old_content and 'DI_EVAL_RES_FILE' not in new_content:
        dsagent_additions.append('DI_EVAL_RES_FILE = EXAMPLE_DATA_PATH / "exp_bank" / "di_eval_res.json"')
    
    if 'DA_EVAL_RES_PATH' in old_content and 'DA_EVAL_RES_PATH' not in new_content:
        dsagent_additions.append('DA_EVAL_RES_PATH = EXAMPLE_DATA_PATH / "exp_bank"')
    
    if dsagent_additions:
        # 追加到文件末尾
        with open(new_file, 'a', encoding='utf-8') as f:
            f.write('\n\n')
            f.write('\n'.join(dsagent_additions))
            f.write('\n')
        print(f"    ✓ 已添加 {len(dsagent_additions)} 个DSAgent常量")

print("✓ 配置文件合并完成")
print()

# 步骤6: 验证关键文件
print("步骤 6: 验证关键文件...")
critical_files = [
    "roles/ds_agent/ds_agent_stream.py",
    "actions/ds_agent/retrieval_exp.py",
    "rag/engines/customSolutionSamplesGenerate.py",
    "const.py",
]

all_exist = True
for file_path in critical_files:
    full_path = old_metagpt / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path}")
        all_exist = False

print()

if all_exist:
    print("=" * 80)
    print("✓ 升级成功完成！")
    print("=" * 80)
    print()
    print("已完成:")
    print("  1. 下载最新版 MetaGPT")
    print("  2. 保留所有 DSAgent 特有功能")
    print("  3. 合并配置文件")
    print("  4. 验证关键文件")
    print()
    print("备份位置:", backup_dir)
    print()
    print("建议:")
    print("  1. 测试 DSAgent 功能是否正常")
    print("  2. 如有问题，可从备份恢复")
    print("  3. 运行: python test_metagpt_integration.py")
else:
    print("=" * 80)
    print("⚠ 升级完成，但有文件缺失")
    print("=" * 80)
    print("请检查上述缺失的文件")

print()
print("清理临时文件...")
temp_dir = project_root / "MetaGPT-latest"
if temp_dir.exists():
    shutil.rmtree(temp_dir)
print("✓ 清理完成")
print()
