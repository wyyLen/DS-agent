# file_service.py
import os
import uuid
import logging
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel

# 配置日志
logger = logging.getLogger(__name__)


class FileInfo(BaseModel):
    file_id: str
    file_name: str
    file_size: int
    mime_type: str
    upload_time: str
    session_id: str
    file_path: str


class FileService:
    """文件服务类，处理文件的上传、获取、删除等操作"""

    def __init__(self, upload_dir: str = "uploads"):
        """
        初始化文件服务

        Args:
            upload_dir: 文件上传存储的目录
        """
        self.upload_dir = upload_dir
        self._ensure_upload_dir()
        self.files_db: Dict[str, FileInfo] = {}  # 简单内存数据库，生产环境应使用真实数据库

    def _ensure_upload_dir(self):
        """确保上传目录存在"""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)
            logger.info(f"创建上传目录: {self.upload_dir}")

    def _get_session_dir(self, session_id: str) -> str:
        """获取会话专属的文件目录"""
        session_dir = os.path.join(self.upload_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)
        return session_dir

    async def save_file(self, file: UploadFile, session_id: str) -> FileInfo:
        """
        保存上传的文件

        Args:
            file: 上传的文件
            session_id: 会话ID

        Returns:
            文件信息
        """
        try:
            # 生成文件ID和保存路径
            file_id = str(uuid.uuid4())
            session_dir = self._get_session_dir(session_id)

            # 提取文件扩展名并构建安全的文件名
            original_filename = file.filename
            file_ext = os.path.splitext(original_filename)[1] if original_filename else ""
            safe_filename = f"{file_id}{file_ext}"
            file_path = os.path.join(session_dir, safe_filename)

            # 保存文件
            file_size = 0
            with open(file_path, "wb") as f:
                # 分块读取并写入文件
                while content := await file.read(1024 * 1024):  # 每次读取1MB
                    f.write(content)
                    file_size += len(content)

            # 创建文件信息
            file_info = FileInfo(
                file_id=file_id,
                file_name=original_filename,
                file_size=file_size,
                mime_type=file.content_type or "application/octet-stream",
                upload_time=datetime.now().isoformat(),
                session_id=session_id,
                file_path=file_path
            )

            # 保存到内存数据库
            self.files_db[file_id] = file_info
            logger.info(f"文件上传成功: {file_info.file_name}, ID: {file_id}")

            return file_info

        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

    def get_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的所有文件

        Args:
            session_id: 会话ID

        Returns:
            文件信息列表
        """
        session_files = []
        for file_id, file_info in self.files_db.items():
            if file_info.session_id == session_id:
                # 排除敏感字段
                session_files.append({
                    "file_id": file_info.file_id,
                    "file_name": file_info.file_name,
                    "file_size": file_info.file_size,
                    "mime_type": file_info.mime_type,
                    "upload_time": file_info.upload_time
                })

        return session_files

    def get_file_info(self, file_id: str) -> Optional[FileInfo]:
        """
        获取文件信息
        Args:
            file_id: 文件ID

        Returns:
            文件信息或None
        """
        return self.files_db.get(file_id)

    def delete_file(self, file_id: str, session_id: str) -> bool:
        """
        删除文件
        Args:
            file_id: 文件ID
            session_id: 会话ID (用于权限验证)

        Returns:
            删除成功返回True，否则返回False
        """
        file_info = self.files_db.get(file_id)
        if not file_info:
            logger.warning(f"尝试删除不存在的文件: {file_id}")
            return False
        # 验证会话ID
        if file_info.session_id != session_id:
            logger.warning(f"会话 {session_id} 尝试删除非自己的文件 {file_id}")
            return False
        try:
            # 删除物理文件
            if os.path.exists(file_info.file_path):
                os.remove(file_info.file_path)
            # 从数据库中移除
            del self.files_db[file_id]
            logger.info(f"文件已删除: {file_id}")
            return True

        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}")
            return False

    def get_file_path(self, file_id: str, session_id: str) -> Optional[str]:
        """
        获取文件路径
        Args:
            file_id: 文件ID
            session_id: 会话ID (用于权限验证)
        Returns:
            文件路径或None
        """
        file_info = self.files_db.get(file_id)
        if not file_info or file_info.session_id != session_id:
            return None

        return file_info.file_path if os.path.exists(file_info.file_path) else None

    def cleanup_session(self, session_id: str) -> int:
        """
        清理会话的所有文件
        Args:
            session_id: 会话ID
        Returns:
            删除的文件数量
        """
        session_dir = os.path.join(self.upload_dir, session_id)
        deleted_count = 0

        # 删除数据库中的记录
        file_ids_to_delete = []
        for file_id, file_info in self.files_db.items():
            if file_info.session_id == session_id:
                file_ids_to_delete.append(file_id)
                deleted_count += 1

        for file_id in file_ids_to_delete:
            del self.files_db[file_id]

        # 删除整个会话目录
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

        logger.info(f"清理会话 {session_id} 的所有文件，共 {deleted_count} 个")
        return deleted_count
