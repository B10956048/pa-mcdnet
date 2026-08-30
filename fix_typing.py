# fix_typing.py
"""
為 Python 3.10 添加缺失的 typing 類型支援
修復 Self 和 Never 類型問題
"""
import sys
import typing

def patch_typing_missing_types():
    """添加缺失的類型到 typing 模組"""
    success_count = 0
    
    try:
        # 修復 Self
        if not hasattr(typing, 'Self'):
            from typing_extensions import Self
            typing.Self = Self
            if hasattr(typing, '__all__') and 'Self' not in typing.__all__:
                typing.__all__.append('Self')
            print("修復: Self 已添加到 typing 模組")
            success_count += 1
        
        # 修復 Never  
        if not hasattr(typing, 'Never'):
            try:
                from typing_extensions import Never
                typing.Never = Never
                if hasattr(typing, '__all__') and 'Never' not in typing.__all__:
                    typing.__all__.append('Never')
                print("修復: Never 已添加到 typing 模組")
                success_count += 1
            except ImportError:
                # 如果 typing_extensions 沒有 Never，創建一個簡單版本
                typing.Never = type('Never', (), {
                    '__module__': 'typing',
                    '__qualname__': 'Never'
                })
                if hasattr(typing, '__all__') and 'Never' not in typing.__all__:
                    typing.__all__.append('Never')
                print("修復: Never 已創建並添加到 typing 模組")
                success_count += 1
        
        # 檢查其他可能缺失的類型
        missing_types = []
        for type_name in ['Unpack', 'Required', 'NotRequired', 'ReadOnly']:
            if not hasattr(typing, type_name):
                try:
                    from typing_extensions import __dict__ as te_dict
                    if type_name in te_dict:
                        setattr(typing, type_name, te_dict[type_name])
                        if hasattr(typing, '__all__') and type_name not in typing.__all__:
                            typing.__all__.append(type_name)
                        print(f"修復: {type_name} 已添加到 typing 模組")
                        success_count += 1
                    else:
                        missing_types.append(type_name)
                except ImportError:
                    missing_types.append(type_name)
        
        if missing_types:
            print(f"注意: 以下類型仍然缺失: {missing_types}")
        
        print(f"typing 模組修復完成，成功修復 {success_count} 個類型")
        return True
        
    except Exception as e:
        print(f"錯誤: 無法修復 typing 模組: {e}")
        return False

# 執行修復
if __name__ == "__main__":
    patch_typing_missing_types()
else:
    # 當被導入時自動執行修復
    patch_typing_missing_types()