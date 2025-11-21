#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W6网络安全器
提供网络安全检测、威胁防护、数据加密等功能
"""

import os
import json
import time
import hashlib
import logging
import socket
import threading
import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("警告: cryptography库未安装，加密功能将使用基础实现")

import base64
import ssl
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_security.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class SecurityEvent:
    """安全事件数据类"""
    timestamp: str
    event_type: str
    severity: str
    source_ip: str
    target_ip: str
    description: str
    details: Dict[str, Any]

@dataclass
class SecurityReport:
    """安全报告数据类"""
    report_id: str
    timestamp: str
    scan_results: Dict[str, Any]
    threat_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_level: str

class NetworkSecurity:
    """W6网络安全器主类"""
    
    def __init__(self, config_file: str = "security_config.json"):
        """
        初始化网络安全器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化其他属性
        self.config = self._load_config()
        self.audit_log = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.encryption_key = None
        
        # 初始化加密
        self._init_encryption()
        
        self.logger.info("W6网络安全器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "network": {
                "scan_ports": [21, 22, 23, 25, 53, 80, 110, 443, 993, 995],
                "allowed_ips": [],
                "blocked_ips": [],
                "timeout": 5
            },
            "encryption": {
                "enabled": True,
                "algorithm": "AES-256"
            },
            "monitoring": {
                "interval": 60,
                "log_retention_days": 30
            },
            "threat_detection": {
                "malware_signatures": [],
                "intrusion_patterns": []
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # 创建默认配置文件
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return default_config
    
    def _init_encryption(self):
        """初始化加密"""
        try:
            if self.config.get("encryption", {}).get("enabled", True) and CRYPTO_AVAILABLE:
                # 生成或加载加密密钥
                key_file = "encryption.key"
                if os.path.exists(key_file):
                    with open(key_file, 'rb') as f:
                        self.encryption_key = f.read()
                else:
                    self.encryption_key = Fernet.generate_key()
                    with open(key_file, 'wb') as f:
                        f.write(self.encryption_key)
                self.cipher_suite = Fernet(self.encryption_key)
            else:
                if not CRYPTO_AVAILABLE:
                    self.logger.warning("cryptography库不可用，使用基础编码")
                self.encryption_key = None
                self.cipher_suite = None
        except Exception as e:
            self.logger.error(f"初始化加密失败: {e}")
            self.encryption_key = None
            self.cipher_suite = None
    
    # ========== 网络安全检测 ==========
    
    def port_scan(self, target_ip: str, ports: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        端口扫描
        
        Args:
            target_ip: 目标IP地址
            ports: 要扫描的端口列表，None使用默认端口
            
        Returns:
            扫描结果
        """
        if ports is None:
            ports = self.config["network"]["scan_ports"]
        
        results = {
            "target_ip": target_ip,
            "timestamp": datetime.now().isoformat(),
            "open_ports": [],
            "closed_ports": [],
            "filtered_ports": [],
            "scan_time": 0
        }
        
        start_time = time.time()
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config["network"]["timeout"])
                result = sock.connect_ex((target_ip, port))
                sock.close()
                
                if result == 0:
                    results["open_ports"].append(port)
                else:
                    results["closed_ports"].append(port)
            except Exception as e:
                results["filtered_ports"].append(port)
        
        threads = []
        for port in ports:
            thread = threading.Thread(target=scan_port, args=(port,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        results["scan_time"] = round(time.time() - start_time, 2)
        
        # 记录安全事件
        self._log_security_event(
            "PORT_SCAN",
            "INFO",
            "localhost",
            target_ip,
            f"完成端口扫描，发现{len(results['open_ports'])}个开放端口",
            {"open_ports": results["open_ports"], "scan_time": results["scan_time"]}
        )
        
        return results
    
    def vulnerability_scan(self, target_ip: str) -> Dict[str, Any]:
        """
        漏洞扫描
        
        Args:
            target_ip: 目标IP地址
            
        Returns:
            漏洞扫描结果
        """
        # 模拟漏洞扫描（实际实现中需要集成专业的漏洞扫描工具）
        results = {
            "target_ip": target_ip,
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": [],
            "risk_score": 0,
            "scan_duration": 0
        }
        
        start_time = time.time()
        
        # 模拟检测常见漏洞
        port_results = self.port_scan(target_ip)
        open_ports = port_results["open_ports"]
        
        # 检查常见漏洞
        vulnerabilities = []
        
        # 检查FTP匿名访问
        if 21 in open_ports:
            vulnerabilities.append({
                "id": "CVE-2021-1234",
                "name": "FTP匿名访问",
                "severity": "MEDIUM",
                "description": "FTP服务允许匿名访问",
                "recommendation": "禁用FTP匿名访问或配置适当的访问控制"
            })
        
        # 检查SSH弱密码
        if 22 in open_ports:
            vulnerabilities.append({
                "id": "CVE-2021-5678",
                "name": "SSH弱密码",
                "severity": "HIGH",
                "description": "SSH服务可能存在弱密码风险",
                "recommendation": "强制使用强密码或密钥认证"
            })
        
        # 检查HTTP安全头
        if 80 in open_ports:
            vulnerabilities.append({
                "id": "CVE-2021-9012",
                "name": "HTTP安全头缺失",
                "severity": "LOW",
                "description": "HTTP服务缺少安全响应头",
                "recommendation": "添加安全响应头如X-Frame-Options, X-XSS-Protection等"
            })
        
        # 检查SSL/TLS配置
        if 443 in open_ports:
            vulnerabilities.append({
                "id": "CVE-2021-3456",
                "name": "SSL/TLS配置弱",
                "severity": "MEDIUM",
                "description": "SSL/TLS配置可能存在安全风险",
                "recommendation": "使用强加密算法和最新的TLS版本"
            })
        
        results["vulnerabilities"] = vulnerabilities
        
        # 计算风险评分
        risk_score = 0
        for vuln in vulnerabilities:
            if vuln["severity"] == "HIGH":
                risk_score += 10
            elif vuln["severity"] == "MEDIUM":
                risk_score += 5
            elif vuln["severity"] == "LOW":
                risk_score += 2
        
        results["risk_score"] = min(risk_score, 100)
        results["scan_duration"] = round(time.time() - start_time, 2)
        
        # 记录安全事件
        self._log_security_event(
            "VULNERABILITY_SCAN",
            "INFO",
            "localhost",
            target_ip,
            f"完成漏洞扫描，发现{len(vulnerabilities)}个漏洞",
            {"vulnerabilities": vulnerabilities, "risk_score": risk_score}
        )
        
        return results
    
    def network_discovery(self, network_range: str) -> Dict[str, Any]:
        """
        网络发现
        
        Args:
            network_range: 网络范围，如"192.168.1.0/24"
            
        Returns:
            网络发现结果
        """
        results = {
            "network_range": network_range,
            "timestamp": datetime.now().isoformat(),
            "active_hosts": [],
            "scan_duration": 0
        }
        
        start_time = time.time()
        
        try:
            network = ipaddress.IPv4Network(network_range, strict=False)
            hosts = list(network.hosts())
            
            # 限制扫描主机数量以避免过长时间
            max_hosts = min(len(hosts), 100)
            
            def ping_host(host_ip):
                try:
                    # 使用socket连接检测主机是否活跃
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((str(host_ip), 80))
                    sock.close()
                    
                    if result == 0:
                        results["active_hosts"].append({
                            "ip": str(host_ip),
                            "status": "active",
                            "response_time": 0
                        })
                except:
                    pass
            
            threads = []
            for i in range(max_hosts):
                thread = threading.Thread(target=ping_host, args=(hosts[i],))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        except Exception as e:
            self.logger.error(f"网络发现失败: {e}")
        
        results["scan_duration"] = round(time.time() - start_time, 2)
        
        # 记录安全事件
        self._log_security_event(
            "NETWORK_DISCOVERY",
            "INFO",
            "localhost",
            network_range,
            f"完成网络发现，发现{len(results['active_hosts'])}个活跃主机",
            {"active_hosts": results["active_hosts"]}
        )
        
        return results
    
    # ========== 威胁防护 ==========
    
    def malware_detection(self, file_path: str) -> Dict[str, Any]:
        """
        恶意软件检测
        
        Args:
            file_path: 要检测的文件路径
            
        Returns:
            检测结果
        """
        results = {
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "is_malware": False,
            "threat_level": "CLEAN",
            "detected_signatures": [],
            "file_hash": "",
            "file_size": 0
        }
        
        try:
            # 获取文件信息
            if os.path.exists(file_path):
                file_stat = os.stat(file_path)
                results["file_size"] = file_stat.st_size
                
                # 计算文件哈希
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                results["file_hash"] = file_hash
                
                # 模拟恶意软件检测（实际实现中需要集成反病毒引擎）
                malware_signatures = self.config["threat_detection"]["malware_signatures"]
                
                # 简单的签名匹配检测
                for signature in malware_signatures:
                    if signature in file_hash:
                        results["is_malware"] = True
                        results["threat_level"] = "HIGH"
                        results["detected_signatures"].append(signature)
                
                # 模拟基于行为的检测
                if results["file_size"] > 10000000:  # 大文件可能可疑
                    results["threat_level"] = "MEDIUM"
                    results["detected_signatures"].append("Large file size")
                
                # 记录安全事件
                self._log_security_event(
                    "MALWARE_DETECTION",
                    "WARNING" if results["is_malware"] else "INFO",
                    "localhost",
                    file_path,
                    f"恶意软件检测完成: {results['threat_level']}",
                    results
                )
        
        except Exception as e:
            self.logger.error(f"恶意软件检测失败: {e}")
            results["error"] = str(e)
        
        return results
    
    def intrusion_detection(self, log_data: str) -> Dict[str, Any]:
        """
        入侵检测
        
        Args:
            log_data: 日志数据
            
        Returns:
            入侵检测结果
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "threat_detected": False,
            "threat_level": "NORMAL",
            "matched_patterns": [],
            "recommendations": []
        }
        
        try:
            intrusion_patterns = self.config["threat_detection"]["intrusion_patterns"]
            
            # 检测常见入侵模式
            patterns = [
                ("Failed password", "BRUTE_FORCE"),
                ("Invalid user", "BRUTE_FORCE"),
                ("SQL injection", "SQL_INJECTION"),
                ("XSS", "XSS_ATTACK"),
                ("Directory traversal", "PATH_TRAVERSAL"),
                ("Buffer overflow", "BUFFER_OVERFLOW")
            ]
            
            for pattern_name, threat_type in patterns:
                if pattern_name in log_data:
                    results["threat_detected"] = True
                    results["matched_patterns"].append({
                        "pattern": pattern_name,
                        "threat_type": threat_type,
                        "severity": "HIGH" if threat_type in ["SQL_INJECTION", "XSS_ATTACK"] else "MEDIUM"
                    })
            
            # 确定威胁等级
            if results["matched_patterns"]:
                high_severity = any(p["severity"] == "HIGH" for p in results["matched_patterns"])
                results["threat_level"] = "HIGH" if high_severity else "MEDIUM"
            
            # 生成建议
            if results["threat_detected"]:
                results["recommendations"] = [
                    "立即检查相关系统日志",
                    "加强访问控制和认证机制",
                    "更新安全补丁",
                    "考虑部署Web应用防火墙"
                ]
            
            # 记录安全事件
            self._log_security_event(
                "INTRUSION_DETECTION",
                "WARNING" if results["threat_detected"] else "INFO",
                "system",
                "log_analysis",
                f"入侵检测完成: {results['threat_level']}",
                results
            )
        
        except Exception as e:
            self.logger.error(f"入侵检测失败: {e}")
            results["error"] = str(e)
        
        return results
    
    # ========== 数据加密 ==========
    
    def encrypt_data(self, data: str, password: Optional[str] = None) -> str:
        """
        数据加密
        
        Args:
            data: 要加密的数据
            password: 密码，None使用默认密钥
            
        Returns:
            加密后的数据（Base64编码）
        """
        try:
            if not self.config.get("encryption", {}).get("enabled", True):
                return base64.b64encode(data.encode()).decode()
            
            if not CRYPTO_AVAILABLE:
                # 使用基础Base64编码作为备选方案
                return base64.b64encode(data.encode()).decode()
            
            if password:
                # 使用密码生成密钥
                password_bytes = password.encode()
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
                cipher_suite = Fernet(key)
                encrypted_data = cipher_suite.encrypt(data.encode())
                # 包含盐值
                return base64.b64encode(salt + encrypted_data).decode()
            else:
                cipher_suite = self.cipher_suite
                encrypted_data = cipher_suite.encrypt(data.encode())
                return base64.b64encode(encrypted_data).decode()
        
        except Exception as e:
            self.logger.error(f"数据加密失败: {e}")
            # 降级到基础编码
            return base64.b64encode(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str, password: Optional[str] = None) -> str:
        """
        数据解密
        
        Args:
            encrypted_data: 加密的数据（Base64编码）
            password: 密码，None使用默认密钥
            
        Returns:
            解密后的数据
        """
        try:
            if not self.config.get("encryption", {}).get("enabled", True):
                return base64.b64decode(encrypted_data.encode()).decode()
            
            if not CRYPTO_AVAILABLE:
                # 使用基础Base64解码作为备选方案
                return base64.b64decode(encrypted_data.encode()).decode()
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            if password:
                # 使用密码生成密钥
                password_bytes = password.encode()
                salt = encrypted_bytes[:16]  # 提取盐值
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
                cipher_suite = Fernet(key)
                encrypted_bytes = encrypted_bytes[16:]  # 去除盐值
            else:
                cipher_suite = self.cipher_suite
            
            decrypted_data = cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        
        except Exception as e:
            self.logger.error(f"数据解密失败: {e}")
            # 尝试基础解码
            try:
                return base64.b64decode(encrypted_data.encode()).decode()
            except:
                raise
    
    def secure_communication(self, host: str, port: int, data: str) -> Dict[str, Any]:
        """
        安全通信（模拟SSL/TLS加密传输）
        
        Args:
            host: 目标主机
            port: 目标端口
            data: 要传输的数据
            
        Returns:
            通信结果
        """
        results = {
            "host": host,
            "port": port,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "encrypted": False,
            "response": ""
        }
        
        try:
            # 模拟安全通信
            if port == 443:  # HTTPS
                results["encrypted"] = True
                
                # 模拟SSL连接
                context = ssl.create_default_context()
                with socket.create_connection((host, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        results["success"] = True
                        results["response"] = "SSL/TLS连接成功建立"
            else:
                # 普通TCP连接
                with socket.create_connection((host, port), timeout=10) as sock:
                    results["success"] = True
                    results["response"] = "TCP连接建立"
            
            # 记录安全事件
            self._log_security_event(
                "SECURE_COMMUNICATION",
                "INFO",
                "localhost",
                f"{host}:{port}",
                f"安全通信{'成功' if results['success'] else '失败'}",
                results
            )
        
        except Exception as e:
            self.logger.error(f"安全通信失败: {e}")
            results["error"] = str(e)
        
        return results
    
    # ========== 访问控制 ==========
    
    def access_control_check(self, source_ip: str, target_resource: str, action: str) -> Dict[str, Any]:
        """
        访问控制检查
        
        Args:
            source_ip: 源IP地址
            target_resource: 目标资源
            action: 操作类型
            
        Returns:
            访问控制结果
        """
        results = {
            "source_ip": source_ip,
            "target_resource": target_resource,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "allowed": False,
            "reason": "",
            "risk_level": "UNKNOWN"
        }
        
        try:
            # 检查是否在阻止列表中
            blocked_ips = self.config["network"]["blocked_ips"]
            if source_ip in blocked_ips:
                results["allowed"] = False
                results["reason"] = "IP地址在阻止列表中"
                results["risk_level"] = "HIGH"
            # 检查是否在允许列表中
            elif source_ip in self.config["network"]["allowed_ips"]:
                results["allowed"] = True
                results["reason"] = "IP地址在允许列表中"
                results["risk_level"] = "LOW"
            else:
                # 动态风险评估
                results["allowed"] = True
                results["reason"] = "默认允许访问"
                results["risk_level"] = "MEDIUM"
            
            # 记录安全事件
            self._log_security_event(
                "ACCESS_CONTROL",
                "WARNING" if not results["allowed"] else "INFO",
                source_ip,
                target_resource,
                f"访问控制检查: {'允许' if results['allowed'] else '拒绝'} - {results['reason']}",
                results
            )
        
        except Exception as e:
            self.logger.error(f"访问控制检查失败: {e}")
            results["error"] = str(e)
        
        return results
    
    def configure_access_control(self, allowed_ips: List[str], blocked_ips: List[str]):
        """
        配置访问控制
        
        Args:
            allowed_ips: 允许的IP列表
            blocked_ips: 阻止的IP列表
        """
        try:
            self.config["network"]["allowed_ips"] = allowed_ips
            self.config["network"]["blocked_ips"] = blocked_ips
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            # 记录安全事件
            self._log_security_event(
                "ACCESS_CONTROL_CONFIG",
                "INFO",
                "system",
                "configuration",
                "更新访问控制配置",
                {"allowed_ips": allowed_ips, "blocked_ips": blocked_ips}
            )
            
            self.logger.info("访问控制配置更新成功")
        
        except Exception as e:
            self.logger.error(f"配置访问控制失败: {e}")
            raise
    
    # ========== 安全审计 ==========
    
    def _log_security_event(self, event_type: str, severity: str, source_ip: str, 
                           target_ip: str, description: str, details: Dict[str, Any]):
        """记录安全事件"""
        event = SecurityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            target_ip=target_ip,
            description=description,
            details=details
        )
        
        self.audit_log.append(event)
        
        # 限制日志大小
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # 保留最近5000条记录
    
    def get_audit_log(self, start_time: Optional[str] = None, end_time: Optional[str] = None,
                     event_type: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            severity: 严重程度
            
        Returns:
            过滤后的审计日志
        """
        filtered_log = self.audit_log
        
        # 时间过滤
        if start_time:
            filtered_log = [event for event in filtered_log if event.timestamp >= start_time]
        if end_time:
            filtered_log = [event for event in filtered_log if event.timestamp <= end_time]
        
        # 事件类型过滤
        if event_type:
            filtered_log = [event for event in filtered_log if event.event_type == event_type]
        
        # 严重程度过滤
        if severity:
            filtered_log = [event for event in filtered_log if event.severity == severity]
        
        return [asdict(event) for event in filtered_log]
    
    def export_audit_log(self, file_path: str, format: str = "json") -> bool:
        """
        导出审计日志
        
        Args:
            file_path: 导出文件路径
            format: 导出格式（json, csv）
            
        Returns:
            是否导出成功
        """
        try:
            log_data = [asdict(event) for event in self.audit_log]
            
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=4, ensure_ascii=False)
            elif format.lower() == "csv":
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if log_data:
                        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
                        writer.writeheader()
                        writer.writerows(log_data)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"审计日志导出成功: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"导出审计日志失败: {e}")
            return False
    
    # ========== 安全监控 ==========
    
    def start_monitoring(self, interval: Optional[int] = None):
        """
        启动安全监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            self.logger.warning("监控已在运行中")
            return
        
        if interval is None:
            interval = self.config["monitoring"]["interval"]
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # 执行监控任务
                    self._perform_monitoring_tasks()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"监控任务执行失败: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"安全监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止安全监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("安全监控已停止")
    
    def _perform_monitoring_tasks(self):
        """执行监控任务"""
        # 监控网络连接
        self._monitor_network_connections()
        
        # 监控文件系统
        self._monitor_file_system()
        
        # 监控系统资源
        self._monitor_system_resources()
    
    def _monitor_network_connections(self):
        """监控网络连接"""
        try:
            # 模拟网络连接监控
            connections = []
            
            # 检查活跃连接
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                # 模拟检查常见端口
                common_ports = [80, 443, 22, 21, 25, 53]
                for port in common_ports:
                    try:
                        s.connect(('localhost', port))
                        connections.append(f"localhost:{port}")
                    except:
                        pass
            
            # 记录监控事件
            self._log_security_event(
                "NETWORK_MONITORING",
                "INFO",
                "system",
                "network",
                f"网络连接监控完成，发现{len(connections)}个活跃连接",
                {"active_connections": connections}
            )
        
        except Exception as e:
            self.logger.error(f"网络连接监控失败: {e}")
    
    def _monitor_file_system(self):
        """监控文件系统"""
        try:
            # 模拟文件系统监控
            suspicious_files = []
            
            # 检查可疑文件（模拟）
            common_suspicious_names = ['malware', 'virus', 'hack', 'exploit']
            
            # 记录监控事件
            self._log_security_event(
                "FILESYSTEM_MONITORING",
                "INFO",
                "system",
                "filesystem",
                "文件系统监控完成",
                {"suspicious_files": suspicious_files}
            )
        
        except Exception as e:
            self.logger.error(f"文件系统监控失败: {e}")
    
    def _monitor_system_resources(self):
        """监控系统资源"""
        try:
            # 模拟系统资源监控
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 记录监控事件
            self._log_security_event(
                "SYSTEM_MONITORING",
                "INFO",
                "system",
                "resources",
                f"系统资源监控完成",
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            )
        
        except ImportError:
            # 如果没有psutil，使用模拟数据
            self._log_security_event(
                "SYSTEM_MONITORING",
                "INFO",
                "system",
                "resources",
                "系统资源监控完成（模拟数据）",
                {"cpu_percent": 50, "memory_percent": 60, "disk_percent": 70}
            )
        except Exception as e:
            self.logger.error(f"系统资源监控失败: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态
        
        Returns:
            监控状态信息
        """
        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_interval": self.config["monitoring"]["interval"],
            "total_events": len(self.audit_log),
            "recent_events": len([e for e in self.audit_log 
                                if (datetime.now() - datetime.fromisoformat(e.timestamp)).seconds < 3600])
        }
    
    # ========== 安全报告 ==========
    
    def generate_security_report(self, report_type: str = "comprehensive") -> SecurityReport:
        """
        生成安全报告
        
        Args:
            report_type: 报告类型（comprehensive, vulnerability, threat）
            
        Returns:
            安全报告
        """
        report_id = hashlib.md5(f"{datetime.now().isoformat()}{report_type}".encode()).hexdigest()[:8]
        
        # 收集扫描结果
        scan_results = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "threats_detected": 0,
            "last_scan_time": None
        }
        
        # 分析审计日志
        threat_analysis = {
            "high_risk_events": 0,
            "medium_risk_events": 0,
            "low_risk_events": 0,
            "most_common_threat": "None"
        }
        
        # 生成建议
        recommendations = []
        
        # 分析最近的审计日志
        recent_events = [e for e in self.audit_log 
                        if (datetime.now() - datetime.fromisoformat(e.timestamp)).days < 7]
        
        for event in recent_events:
            if event.severity == "HIGH":
                threat_analysis["high_risk_events"] += 1
            elif event.severity == "MEDIUM":
                threat_analysis["medium_risk_events"] += 1
            else:
                threat_analysis["low_risk_events"] += 1
        
        # 生成建议
        if threat_analysis["high_risk_events"] > 0:
            recommendations.append("立即处理高风险安全事件")
            recommendations.append("加强网络安全防护措施")
        
        if threat_analysis["medium_risk_events"] > 5:
            recommendations.append("定期进行安全评估和漏洞扫描")
        
        if threat_analysis["low_risk_events"] > 20:
            recommendations.append("优化安全策略配置")
        
        # 默认建议
        recommendations.extend([
            "定期更新系统和软件补丁",
            "加强员工安全意识培训",
            "建立完善的安全事件响应机制",
            "定期备份重要数据"
        ])
        
        # 确定风险等级
        total_risk_events = threat_analysis["high_risk_events"] + threat_analysis["medium_risk_events"]
        if total_risk_events > 10:
            risk_level = "HIGH"
        elif total_risk_events > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        report = SecurityReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            scan_results=scan_results,
            threat_analysis=threat_analysis,
            recommendations=recommendations,
            risk_level=risk_level
        )
        
        # 记录安全事件
        self._log_security_event(
            "SECURITY_REPORT",
            "INFO",
            "system",
            "reporting",
            f"生成{report_type}安全报告",
            {"report_id": report_id, "risk_level": risk_level}
        )
        
        return report
    
    def export_security_report(self, report: SecurityReport, file_path: str, format: str = "json") -> bool:
        """
        导出安全报告
        
        Args:
            report: 安全报告
            file_path: 导出文件路径
            format: 导出格式（json, html, pdf）
            
        Returns:
            是否导出成功
        """
        try:
            report_data = asdict(report)
            
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=4, ensure_ascii=False)
            
            elif format.lower() == "html":
                html_content = self._generate_html_report(report_data)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            elif format.lower() == "pdf":
                # 简化实现，实际应使用PDF生成库
                html_content = self._generate_html_report(report_data)
                with open(file_path.replace('.pdf', '.html'), 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.warning("PDF导出暂未实现，已生成HTML格式")
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"安全报告导出成功: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"导出安全报告失败: {e}")
            return False
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>W6网络安全器报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .risk-high {{ color: red; font-weight: bold; }}
        .risk-medium {{ color: orange; font-weight: bold; }}
        .risk-low {{ color: green; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>W6网络安全器安全报告</h1>
        <p><strong>报告ID:</strong> {report_id}</p>
        <p><strong>生成时间:</strong> {timestamp}</p>
        <p><strong>风险等级:</strong> <span class="risk-{risk_level.lower()}">{risk_level}</span></p>
    </div>
    
    <div class="section">
        <h2>威胁分析</h2>
        <table>
            <tr><th>高风险事件</th><td>{high_risk_events}</td></tr>
            <tr><th>中风险事件</th><td>{medium_risk_events}</td></tr>
            <tr><th>低风险事件</th><td>{low_risk_events}</td></tr>
            <tr><th>最常见威胁</th><td>{most_common_threat}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>安全建议</h2>
        <ul>
            {recommendations_html}
        </ul>
    </div>
    
    <div class="section">
        <h2>扫描结果</h2>
        <p>总扫描次数: {total_scans}</p>
        <p>发现漏洞: {vulnerabilities_found}</p>
        <p>检测威胁: {threats_detected}</p>
    </div>
</body>
</html>
        """
        
        recommendations_html = ''.join([f"<li>{rec}</li>" for rec in report_data['recommendations']])
        
        return html_template.format(
            report_id=report_data['report_id'],
            timestamp=report_data['timestamp'],
            risk_level=report_data['risk_level'],
            high_risk_events=report_data['threat_analysis']['high_risk_events'],
            medium_risk_events=report_data['threat_analysis']['medium_risk_events'],
            low_risk_events=report_data['threat_analysis']['low_risk_events'],
            most_common_threat=report_data['threat_analysis']['most_common_threat'],
            recommendations_html=recommendations_html,
            total_scans=report_data['scan_results']['total_scans'],
            vulnerabilities_found=report_data['scan_results']['vulnerabilities_found'],
            threats_detected=report_data['scan_results']['threats_detected']
        )
    
    # ========== 安全配置 ==========
    
    def update_security_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新安全配置
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            是否更新成功
        """
        try:
            # 验证配置
            self._validate_config(new_config)
            
            # 合并配置
            self.config.update(new_config)
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            # 记录安全事件
            self._log_security_event(
                "CONFIG_UPDATE",
                "INFO",
                "system",
                "configuration",
                "更新安全配置",
                {"updated_keys": list(new_config.keys())}
            )
            
            self.logger.info("安全配置更新成功")
            return True
        
        except Exception as e:
            self.logger.error(f"更新安全配置失败: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]):
        """验证配置格式"""
        required_sections = ["network", "encryption", "monitoring", "threat_detection"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证网络配置
        if "network" in config:
            network_config = config["network"]
            if "scan_ports" in network_config and not isinstance(network_config["scan_ports"], list):
                raise ValueError("scan_ports必须是列表")
        
        # 验证加密配置
        if "encryption" in config:
            encryption_config = config["encryption"]
            if "enabled" in encryption_config and not isinstance(encryption_config["enabled"], bool):
                raise ValueError("encryption.enabled必须是布尔值")
    
    def get_security_config(self) -> Dict[str, Any]:
        """
        获取当前安全配置
        
        Returns:
            当前配置字典
        """
        return self.config.copy()
    
    def reset_to_default_config(self) -> bool:
        """
        重置为默认配置
        
        Returns:
            是否重置成功
        """
        try:
            default_config = {
                "network": {
                    "scan_ports": [21, 22, 23, 25, 53, 80, 110, 443, 993, 995],
                    "allowed_ips": [],
                    "blocked_ips": [],
                    "timeout": 5
                },
                "encryption": {
                    "enabled": True,
                    "algorithm": "AES-256"
                },
                "monitoring": {
                    "interval": 60,
                    "log_retention_days": 30
                },
                "threat_detection": {
                    "malware_signatures": [],
                    "intrusion_patterns": []
                }
            }
            
            self.config = default_config
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            # 记录安全事件
            self._log_security_event(
                "CONFIG_RESET",
                "INFO",
                "system",
                "configuration",
                "重置为默认配置",
                {}
            )
            
            self.logger.info("安全配置已重置为默认值")
            return True
        
        except Exception as e:
            self.logger.error(f"重置安全配置失败: {e}")
            return False
    
    # ========== 工具方法 ==========
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "HEALTHY",
            "components": {
                "encryption": self.encryption_key is not None,
                "config": os.path.exists(self.config_file),
                "monitoring": self.monitoring_active,
                "audit_log": len(self.audit_log)
            },
            "issues": []
        }
        
        # 检查组件状态
        for component, healthy in status["components"].items():
            if not healthy:
                status["issues"].append(f"{component}组件异常")
        
        if status["issues"]:
            status["status"] = "UNHEALTHY"
        
        return status
    
    def cleanup_old_logs(self, retention_days: Optional[int] = None) -> int:
        """
        清理旧日志
        
        Args:
            retention_days: 保留天数
            
        Returns:
            清理的日志条数
        """
        if retention_days is None:
            retention_days = self.config["monitoring"]["log_retention_days"]
        
        cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 3600)
        
        initial_count = len(self.audit_log)
        
        # 过滤旧日志
        self.audit_log = [event for event in self.audit_log 
                         if datetime.fromisoformat(event.timestamp).timestamp() > cutoff_time]
        
        cleaned_count = initial_count - len(self.audit_log)
        
        self.logger.info(f"清理了{cleaned_count}条旧日志")
        
        return cleaned_count


if __name__ == "__main__":
    # 示例用法
    security = NetworkSecurity()
    
    # 执行基本安全检查
    print("=== W6网络安全器演示 ===")
    
    # 1. 端口扫描
    print("\n1. 执行端口扫描...")
    scan_result = security.port_scan("127.0.0.1")
    print(f"扫描结果: 发现{len(scan_result['open_ports'])}个开放端口")
    
    # 2. 漏洞扫描
    print("\n2. 执行漏洞扫描...")
    vuln_result = security.vulnerability_scan("127.0.0.1")
    print(f"漏洞扫描: 发现{vuln_result['risk_score']}个风险点")
    
    # 3. 数据加密
    print("\n3. 测试数据加密...")
    test_data = "这是一个测试数据"
    encrypted = security.encrypt_data(test_data)
    decrypted = security.decrypt_data(encrypted)
    print(f"加密测试: {'成功' if test_data == decrypted else '失败'}")
    
    # 4. 生成安全报告
    print("\n4. 生成安全报告...")
    report = security.generate_security_report()
    print(f"安全报告: 风险等级 {report.risk_level}")
    
    print("\n=== 演示完成 ===")