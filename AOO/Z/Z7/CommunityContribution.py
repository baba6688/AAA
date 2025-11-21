"""
Z7社区贡献接口实现
包含贡献管理、审核流程、社区协作等功能的完整实现
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import uuid


class ContributionType(Enum):
    """贡献类型枚举"""
    CODE = "code"  # 代码贡献
    FEATURE = "feature"  # 功能贡献
    DOCUMENTATION = "documentation"  # 文档贡献
    TESTING = "testing"  # 测试贡献
    DESIGN = "design"  # 设计贡献
    BUG_FIX = "bug_fix"  # Bug修复贡献


class ContributionStatus(Enum):
    """贡献状态枚举"""
    PENDING = "pending"  # 待审核
    APPROVED = "approved"  # 已批准
    REJECTED = "rejected"  # 已拒绝
    IN_REVIEW = "in_review"  # 审核中
    MERGED = "merged"  # 已合并
    CLOSED = "closed"  # 已关闭


class UserRole(Enum):
    """用户角色枚举"""
    MEMBER = "member"  # 普通成员
    CONTRIBUTOR = "contributor"  # 贡献者
    MAINTAINER = "maintainer"  # 维护者
    ADMIN = "admin"  # 管理员


class Contribution:
    """贡献数据类"""
    
    def __init__(self, 
                 contribution_id: str,
                 title: str,
                 description: str,
                 contribution_type: ContributionType,
                 contributor_id: str,
                 created_at: datetime = None):
        self.contribution_id = contribution_id
        self.title = title
        self.description = description
        self.contribution_type = contribution_type
        self.contributor_id = contributor_id
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at
        self.status = ContributionStatus.PENDING
        self.reviewers: List[str] = []
        self.approved_by: Optional[str] = None
        self.rejected_by: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        self.merged_at: Optional[datetime] = None
        self.tags: List[str] = []
        self.priority = 0
        self.difficulty = 0
        self.estimated_hours = 0
        self.actual_hours = 0
        self.reward_points = 0
        self.files_changed: List[str] = []
        self.lines_added = 0
        self.lines_removed = 0
        self.comments: List[Dict[str, Any]] = []
        self.dependencies: List[str] = []
        self.milestone: Optional[str] = None
        self.review_comments: List[Dict[str, Any]] = []
        self.test_coverage = 0.0
        self.code_quality_score = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'contribution_id': self.contribution_id,
            'title': self.title,
            'description': self.description,
            'contribution_type': self.contribution_type.value,
            'contributor_id': self.contributor_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'reviewers': self.reviewers,
            'approved_by': self.approved_by,
            'rejected_by': self.rejected_by,
            'rejection_reason': self.rejection_reason,
            'merged_at': self.merged_at.isoformat() if self.merged_at else None,
            'tags': self.tags,
            'priority': self.priority,
            'difficulty': self.difficulty,
            'estimated_hours': self.estimated_hours,
            'actual_hours': self.actual_hours,
            'reward_points': self.reward_points,
            'files_changed': self.files_changed,
            'lines_added': self.lines_added,
            'lines_removed': self.lines_removed,
            'comments': self.comments,
            'dependencies': self.dependencies,
            'milestone': self.milestone,
            'review_comments': self.review_comments,
            'test_coverage': self.test_coverage,
            'code_quality_score': self.code_quality_score
        }


class Contributor:
    """贡献者数据类"""
    
    def __init__(self, 
                 contributor_id: str,
                 username: str,
                 email: str,
                 role: UserRole = UserRole.MEMBER):
        self.contributor_id = contributor_id
        self.username = username
        self.email = email
        self.role = role
        self.join_date = datetime.now()
        self.last_active = datetime.now()
        self.total_contributions = 0
        self.approved_contributions = 0
        self.rejected_contributions = 0
        self.total_reward_points = 0
        self.skills: List[str] = []
        self.bio = ""
        self.avatar_url = ""
        self.location = ""
        self.timezone = "UTC"
        self.preferences: Dict[str, Any] = {}
        self.statistics: Dict[str, Any] = {}
        self.badges: List[str] = []
        self.level = 1
        self.experience_points = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'contributor_id': self.contributor_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'join_date': self.join_date.isoformat(),
            'last_active': self.last_active.isoformat(),
            'total_contributions': self.total_contributions,
            'approved_contributions': self.approved_contributions,
            'rejected_contributions': self.rejected_contributions,
            'total_reward_points': self.total_reward_points,
            'skills': self.skills,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'location': self.location,
            'timezone': self.timezone,
            'preferences': self.preferences,
            'statistics': self.statistics,
            'badges': self.badges,
            'level': self.level,
            'experience_points': self.experience_points
        }


class CommunityContribution:
    """Z7社区贡献接口主类"""
    
    def __init__(self):
        self.contributions: Dict[str, Contribution] = {}
        self.contributors: Dict[str, Contributor] = {}
        self.reviews: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []
        self.collaboration_sessions: List[Dict[str, Any]] = []
        self.rewards: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {}
        
        # 初始化统计数据
        self._init_statistics()

    def _init_statistics(self):
        """初始化统计数据"""
        self.statistics = {
            'total_contributions': 0,
            'approved_contributions': 0,
            'rejected_contributions': 0,
            'pending_contributions': 0,
            'total_contributors': 0,
            'active_contributors': 0,
            'total_reward_points_distributed': 0,
            'average_review_time': 0,
            'contribution_types': {},
            'monthly_contributions': {},
            'top_contributors': []
        }

    # ==================== 贡献管理 ====================
    
    def create_contribution(self, 
                           title: str,
                           description: str,
                           contribution_type: ContributionType,
                           contributor_id: str,
                           **kwargs) -> str:
        """
        创建新的贡献
        
        Args:
            title: 贡献标题
            description: 贡献描述
            contribution_type: 贡献类型
            contributor_id: 贡献者ID
            **kwargs: 其他参数
            
        Returns:
            str: 贡献ID
        """
        contribution_id = str(uuid.uuid4())
        
        contribution = Contribution(
            contribution_id=contribution_id,
            title=title,
            description=description,
            contribution_type=contribution_type,
            contributor_id=contributor_id
        )
        
        # 设置其他属性
        for key, value in kwargs.items():
            if hasattr(contribution, key):
                setattr(contribution, key, value)
        
        self.contributions[contribution_id] = contribution
        
        # 更新统计
        self.statistics['total_contributions'] += 1
        self.statistics['pending_contributions'] += 1
        
        if contributor_id in self.contributors:
            contributor = self.contributors[contributor_id]
            contributor.total_contributions += 1
            contributor.last_active = datetime.now()
        
        return contribution_id

    def update_contribution(self, 
                           contribution_id: str,
                           **updates) -> bool:
        """
        更新贡献信息
        
        Args:
            contribution_id: 贡献ID
            **updates: 更新字段
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        
        for field, value in updates.items():
            if hasattr(contribution, field):
                setattr(contribution, field, value)
        
        contribution.updated_at = datetime.now()
        return True

    def delete_contribution(self, contribution_id: str) -> bool:
        """
        删除贡献
        
        Args:
            contribution_id: 贡献ID
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        
        # 更新统计
        self.statistics['total_contributions'] -= 1
        if contribution.status == ContributionStatus.PENDING:
            self.statistics['pending_contributions'] -= 1
        elif contribution.status == ContributionStatus.APPROVED:
            self.statistics['approved_contributions'] -= 1
        elif contribution.status == ContributionStatus.REJECTED:
            self.statistics['rejected_contributions'] -= 1
        
        del self.contributions[contribution_id]
        return True

    def get_contribution(self, contribution_id: str) -> Optional[Contribution]:
        """
        获取贡献信息
        
        Args:
            contribution_id: 贡献ID
            
        Returns:
            Contribution: 贡献对象
        """
        return self.contributions.get(contribution_id)

    def list_contributions(self, 
                          status: Optional[ContributionStatus] = None,
                          contribution_type: Optional[ContributionType] = None,
                          contributor_id: Optional[str] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[Contribution]:
        """
        列出贡献
        
        Args:
            status: 状态过滤
            contribution_type: 类型过滤
            contributor_id: 贡献者ID过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            List[Contribution]: 贡献列表
        """
        filtered_contributions = list(self.contributions.values())
        
        if status:
            filtered_contributions = [c for c in filtered_contributions if c.status == status]
        
        if contribution_type:
            filtered_contributions = [c for c in filtered_contributions if c.contribution_type == contribution_type]
        
        if contributor_id:
            filtered_contributions = [c for c in filtered_contributions if c.contributor_id == contributor_id]
        
        # 按创建时间倒序排列
        filtered_contributions.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_contributions[offset:offset + limit]

    # ==================== 审核流程 ====================
    
    def submit_for_review(self, contribution_id: str, reviewer_ids: List[str]) -> bool:
        """
        提交贡献进行审核
        
        Args:
            contribution_id: 贡献ID
            reviewer_ids: 审核者ID列表
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        contribution.status = ContributionStatus.IN_REVIEW
        contribution.reviewers = reviewer_ids
        contribution.updated_at = datetime.now()
        
        # 创建审核记录
        review_record = {
            'contribution_id': contribution_id,
            'reviewers': reviewer_ids,
            'submitted_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        self.reviews.append(review_record)
        
        return True

    def approve_contribution(self, 
                           contribution_id: str,
                           reviewer_id: str,
                           comments: str = "") -> bool:
        """
        批准贡献
        
        Args:
            contribution_id: 贡献ID
            reviewer_id: 审核者ID
            comments: 审核评论
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        contribution.status = ContributionStatus.APPROVED
        contribution.approved_by = reviewer_id
        contribution.updated_at = datetime.now()
        
        # 更新统计
        self.statistics['pending_contributions'] -= 1
        self.statistics['approved_contributions'] += 1
        
        # 计算奖励积分
        self._calculate_reward_points(contribution)
        
        # 更新贡献者统计
        if contribution.contributor_id in self.contributors:
            contributor = self.contributors[contribution.contributor_id]
            contributor.approved_contributions += 1
            contributor.total_reward_points += contribution.reward_points
        
        # 添加审核评论
        review_comment = {
            'reviewer_id': reviewer_id,
            'action': 'approved',
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        contribution.review_comments.append(review_comment)
        
        return True

    def reject_contribution(self, 
                          contribution_id: str,
                          reviewer_id: str,
                          reason: str,
                          comments: str = "") -> bool:
        """
        拒绝贡献
        
        Args:
            contribution_id: 贡献ID
            reviewer_id: 审核者ID
            reason: 拒绝原因
            comments: 审核评论
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        contribution.status = ContributionStatus.REJECTED
        contribution.rejected_by = reviewer_id
        contribution.rejection_reason = reason
        contribution.updated_at = datetime.now()
        
        # 更新统计
        self.statistics['pending_contributions'] -= 1
        self.statistics['rejected_contributions'] += 1
        
        # 更新贡献者统计
        if contribution.contributor_id in self.contributors:
            contributor = self.contributors[contribution.contributor_id]
            contributor.rejected_contributions += 1
        
        # 添加审核评论
        review_comment = {
            'reviewer_id': reviewer_id,
            'action': 'rejected',
            'reason': reason,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        contribution.review_comments.append(review_comment)
        
        return True

    def merge_contribution(self, contribution_id: str, merger_id: str) -> bool:
        """
        合并贡献
        
        Args:
            contribution_id: 贡献ID
            merger_id: 合并者ID
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        if contribution.status != ContributionStatus.APPROVED:
            return False
        
        contribution.status = ContributionStatus.MERGED
        contribution.merged_at = datetime.now()
        contribution.updated_at = datetime.now()
        
        return True

    # ==================== 社区协作 ====================
    
    def create_collaboration_session(self, 
                                   contribution_id: str,
                                   initiator_id: str,
                                   participant_ids: List[str],
                                   session_type: str = "review") -> str:
        """
        创建协作会话
        
        Args:
            contribution_id: 贡献ID
            initiator_id: 发起者ID
            participant_ids: 参与者ID列表
            session_type: 会话类型
            
        Returns:
            str: 会话ID
        """
        session_id = str(uuid.uuid4())
        
        session = {
            'session_id': session_id,
            'contribution_id': contribution_id,
            'initiator_id': initiator_id,
            'participant_ids': participant_ids,
            'session_type': session_type,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'messages': []
        }
        
        self.collaboration_sessions.append(session)
        return session_id

    def add_collaboration_message(self, 
                                session_id: str,
                                user_id: str,
                                message: str) -> bool:
        """
        添加协作消息
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            message: 消息内容
            
        Returns:
            bool: 是否成功
        """
        for session in self.collaboration_sessions:
            if session['session_id'] == session_id:
                msg = {
                    'user_id': user_id,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                session['messages'].append(msg)
                return True
        
        return False

    def close_collaboration_session(self, session_id: str) -> bool:
        """
        关闭协作会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功
        """
        for session in self.collaboration_sessions:
            if session['session_id'] == session_id:
                session['status'] = 'closed'
                session['closed_at'] = datetime.now().isoformat()
                return True
        
        return False

    # ==================== 贡献统计 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取贡献统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 更新实时统计
        self._update_statistics()
        return self.statistics.copy()

    def _update_statistics(self):
        """更新统计数据"""
        # 按类型统计
        type_stats = {}
        for contribution in self.contributions.values():
            ctype = contribution.contribution_type.value
            if ctype not in type_stats:
                type_stats[ctype] = 0
            type_stats[ctype] += 1
        self.statistics['contribution_types'] = type_stats
        
        # 按月统计
        monthly_stats = {}
        for contribution in self.contributions.values():
            month_key = contribution.created_at.strftime('%Y-%m')
            if month_key not in monthly_stats:
                monthly_stats[month_key] = 0
            monthly_stats[month_key] += 1
        self.statistics['monthly_contributions'] = monthly_stats
        
        # 活跃贡献者统计
        active_contributors = 0
        thirty_days_ago = datetime.now() - timedelta(days=30)
        for contributor in self.contributors.values():
            if contributor.last_active >= thirty_days_ago:
                active_contributors += 1
        self.statistics['active_contributors'] = active_contributors
        
        # 顶级贡献者
        top_contributors = sorted(
            self.contributors.values(),
            key=lambda x: x.approved_contributions,
            reverse=True
        )[:10]
        self.statistics['top_contributors'] = [
            {
                'username': c.username,
                'approved_contributions': c.approved_contributions,
                'total_reward_points': c.total_reward_points
            }
            for c in top_contributors
        ]

    def get_contributor_statistics(self, contributor_id: str) -> Optional[Dict[str, Any]]:
        """
        获取贡献者统计信息
        
        Args:
            contributor_id: 贡献者ID
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if contributor_id not in self.contributors:
            return None
        
        contributor = self.contributors[contributor_id]
        
        # 获取该贡献者的所有贡献
        contributions = [
            c for c in self.contributions.values()
            if c.contributor_id == contributor_id
        ]
        
        # 按类型统计
        type_stats = {}
        for contribution in contributions:
            ctype = contribution.contribution_type.value
            if ctype not in type_stats:
                type_stats[ctype] = {'total': 0, 'approved': 0, 'rejected': 0}
            type_stats[ctype]['total'] += 1
            if contribution.status == ContributionStatus.APPROVED:
                type_stats[ctype]['approved'] += 1
            elif contribution.status == ContributionStatus.REJECTED:
                type_stats[ctype]['rejected'] += 1
        
        # 按月统计
        monthly_stats = {}
        for contribution in contributions:
            month_key = contribution.created_at.strftime('%Y-%m')
            if month_key not in monthly_stats:
                monthly_stats[month_key] = 0
            monthly_stats[month_key] += 1
        
        return {
            'contributor': contributor.to_dict(),
            'contributions': {
                'total': len(contributions),
                'approved': len([c for c in contributions if c.status == ContributionStatus.APPROVED]),
                'rejected': len([c for c in contributions if c.status == ContributionStatus.REJECTED]),
                'pending': len([c for c in contributions if c.status == ContributionStatus.PENDING]),
                'in_review': len([c for c in contributions if c.status == ContributionStatus.IN_REVIEW])
            },
            'by_type': type_stats,
            'monthly': monthly_stats,
            'average_review_time': self._calculate_average_review_time(contributions),
            'success_rate': len([c for c in contributions if c.status == ContributionStatus.APPROVED]) / len(contributions) if contributions else 0
        }

    def _calculate_average_review_time(self, contributions: List[Contribution]) -> float:
        """计算平均审核时间（小时）"""
        reviewed_contributions = [
            c for c in contributions
            if c.status in [ContributionStatus.APPROVED, ContributionStatus.REJECTED]
        ]
        
        if not reviewed_contributions:
            return 0
        
        total_time = 0
        for contribution in reviewed_contributions:
            review_time = (contribution.updated_at - contribution.created_at).total_seconds() / 3600
            total_time += review_time
        
        return total_time / len(reviewed_contributions)

    # ==================== 贡献者管理 ====================
    
    def register_contributor(self, 
                           username: str,
                           email: str,
                           role: UserRole = UserRole.MEMBER,
                           **kwargs) -> str:
        """
        注册贡献者
        
        Args:
            username: 用户名
            email: 邮箱
            role: 用户角色
            **kwargs: 其他参数
            
        Returns:
            str: 贡献者ID
        """
        contributor_id = str(uuid.uuid4())
        
        contributor = Contributor(
            contributor_id=contributor_id,
            username=username,
            email=email,
            role=role
        )
        
        # 设置其他属性
        for key, value in kwargs.items():
            if hasattr(contributor, key):
                setattr(contributor, key, value)
        
        self.contributors[contributor_id] = contributor
        
        # 更新统计
        self.statistics['total_contributors'] += 1
        
        return contributor_id

    def update_contributor(self, 
                         contributor_id: str,
                         **updates) -> bool:
        """
        更新贡献者信息
        
        Args:
            contributor_id: 贡献者ID
            **updates: 更新字段
            
        Returns:
            bool: 是否成功
        """
        if contributor_id not in self.contributors:
            return False
        
        contributor = self.contributors[contributor_id]
        
        for field, value in updates.items():
            if hasattr(contributor, field):
                setattr(contributor, field, value)
        
        contributor.last_active = datetime.now()
        return True

    def get_contributor(self, contributor_id: str) -> Optional[Contributor]:
        """
        获取贡献者信息
        
        Args:
            contributor_id: 贡献者ID
            
        Returns:
            Contributor: 贡献者对象
        """
        return self.contributors.get(contributor_id)

    def find_contributor_by_username(self, username: str) -> Optional[Contributor]:
        """
        通过用户名查找贡献者
        
        Args:
            username: 用户名
            
        Returns:
            Contributor: 贡献者对象
        """
        for contributor in self.contributors.values():
            if contributor.username == username:
                return contributor
        return None

    def list_contributors(self, 
                        role: Optional[UserRole] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Contributor]:
        """
        列出贡献者
        
        Args:
            role: 角色过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            List[Contributor]: 贡献者列表
        """
        filtered_contributors = list(self.contributors.values())
        
        if role:
            filtered_contributors = [c for c in filtered_contributors if c.role == role]
        
        # 按贡献数量倒序排列
        filtered_contributors.sort(key=lambda x: x.approved_contributions, reverse=True)
        
        return filtered_contributors[offset:offset + limit]

    # ==================== 贡献记录 ====================
    
    def get_contribution_history(self, contribution_id: str) -> List[Dict[str, Any]]:
        """
        获取贡献历史记录
        
        Args:
            contribution_id: 贡献ID
            
        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        if contribution_id not in self.contributions:
            return []
        
        contribution = self.contributions[contribution_id]
        history = []
        
        # 创建记录
        history.append({
            'action': 'created',
            'timestamp': contribution.created_at.isoformat(),
            'details': f'贡献 "{contribution.title}" 已创建'
        })
        
        # 审核记录
        for review in contribution.review_comments:
            history.append({
                'action': 'reviewed',
                'timestamp': review['timestamp'],
                'reviewer_id': review['reviewer_id'],
                'action_type': review['action'],
                'details': review.get('comments', '')
            })
        
        # 状态变更记录
        if contribution.merged_at:
            history.append({
                'action': 'merged',
                'timestamp': contribution.merged_at.isoformat(),
                'details': f'贡献 "{contribution.title}" 已合并'
            })
        
        # 按时间排序
        history.sort(key=lambda x: x['timestamp'])
        
        return history

    def add_comment(self, 
                   contribution_id: str,
                   user_id: str,
                   comment: str,
                   comment_type: str = "general") -> bool:
        """
        添加评论
        
        Args:
            contribution_id: 贡献ID
            user_id: 用户ID
            comment: 评论内容
            comment_type: 评论类型
            
        Returns:
            bool: 是否成功
        """
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        
        comment_obj = {
            'user_id': user_id,
            'comment': comment,
            'comment_type': comment_type,
            'timestamp': datetime.now().isoformat()
        }
        
        contribution.comments.append(comment_obj)
        return True

    # ==================== 贡献奖励 ====================
    
    def _calculate_reward_points(self, contribution: Contribution):
        """计算贡献奖励积分"""
        base_points = {
            ContributionType.CODE: 100,
            ContributionType.FEATURE: 150,
            ContributionType.DOCUMENTATION: 50,
            ContributionType.TESTING: 75,
            ContributionType.DESIGN: 80,
            ContributionType.BUG_FIX: 60
        }
        
        points = base_points.get(contribution.contribution_type, 50)
        
        # 根据难度调整
        difficulty_multiplier = 1 + (contribution.difficulty * 0.2)
        points *= difficulty_multiplier
        
        # 根据优先级调整
        priority_multiplier = 1 + (contribution.priority * 0.1)
        points *= priority_multiplier
        
        # 根据代码质量调整
        quality_bonus = contribution.code_quality_score * 10
        points += quality_bonus
        
        # 根据测试覆盖率调整
        coverage_bonus = contribution.test_coverage * 20
        points += coverage_bonus
        
        # 根据实际工作量调整
        if contribution.estimated_hours > 0:
            efficiency = contribution.estimated_hours / max(contribution.actual_hours, 1)
            points *= min(efficiency, 2.0)  # 最高2倍效率奖励
        
        contribution.reward_points = int(points)

    def create_reward(self, 
                     contributor_id: str,
                     contribution_id: str,
                     reward_type: str,
                     points: int,
                     description: str) -> str:
        """
        创建奖励记录
        
        Args:
            contributor_id: 贡献者ID
            contribution_id: 贡献ID
            reward_type: 奖励类型
            points: 积分数量
            description: 奖励描述
            
        Returns:
            str: 奖励ID
        """
        reward_id = str(uuid.uuid4())
        
        reward = {
            'reward_id': reward_id,
            'contributor_id': contributor_id,
            'contribution_id': contribution_id,
            'reward_type': reward_type,
            'points': points,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        self.rewards.append(reward)
        
        # 更新总积分
        self.statistics['total_reward_points_distributed'] += points
        
        return reward_id

    def get_contributor_rewards(self, contributor_id: str) -> List[Dict[str, Any]]:
        """
        获取贡献者奖励记录
        
        Args:
            contributor_id: 贡献者ID
            
        Returns:
            List[Dict[str, Any]]: 奖励记录列表
        """
        return [
            reward for reward in self.rewards
            if reward['contributor_id'] == contributor_id
        ]

    # ==================== 社区反馈 ====================
    
    def submit_feedback(self, 
                       user_id: str,
                       feedback_type: str,
                       content: str,
                       contribution_id: Optional[str] = None,
                       priority: int = 1) -> str:
        """
        提交社区反馈
        
        Args:
            user_id: 用户ID
            feedback_type: 反馈类型
            content: 反馈内容
            contribution_id: 关联的贡献ID
            priority: 优先级
            
        Returns:
            str: 反馈ID
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = {
            'feedback_id': feedback_id,
            'user_id': user_id,
            'feedback_type': feedback_type,
            'content': content,
            'contribution_id': contribution_id,
            'priority': priority,
            'status': 'open',
            'created_at': datetime.now().isoformat(),
            'responses': []
        }
        
        self.feedback.append(feedback)
        return feedback_id

    def respond_to_feedback(self, 
                          feedback_id: str,
                          responder_id: str,
                          response: str) -> bool:
        """
        回应反馈
        
        Args:
            feedback_id: 反馈ID
            responder_id: 回应者ID
            response: 回应内容
            
        Returns:
            bool: 是否成功
        """
        for feedback in self.feedback:
            if feedback['feedback_id'] == feedback_id:
                response_obj = {
                    'responder_id': responder_id,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
                feedback['responses'].append(response_obj)
                return True
        
        return False

    def close_feedback(self, feedback_id: str) -> bool:
        """
        关闭反馈
        
        Args:
            feedback_id: 反馈ID
            
        Returns:
            bool: 是否成功
        """
        for feedback in self.feedback:
            if feedback['feedback_id'] == feedback_id:
                feedback['status'] = 'closed'
                feedback['closed_at'] = datetime.now().isoformat()
                return True
        
        return False

    def list_feedback(self, 
                     status: Optional[str] = None,
                     feedback_type: Optional[str] = None,
                     limit: int = 100,
                     offset: int = 0) -> List[Dict[str, Any]]:
        """
        列出反馈
        
        Args:
            status: 状态过滤
            feedback_type: 类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            List[Dict[str, Any]]: 反馈列表
        """
        filtered_feedback = self.feedback.copy()
        
        if status:
            filtered_feedback = [f for f in filtered_feedback if f['status'] == status]
        
        if feedback_type:
            filtered_feedback = [f for f in filtered_feedback if f['feedback_type'] == feedback_type]
        
        # 按创建时间倒序排列
        filtered_feedback.sort(key=lambda x: x['created_at'], reverse=True)
        
        return filtered_feedback[offset:offset + limit]

    # ==================== 数据导出 ====================
    
    def export_data(self, format_type: str = "json") -> str:
        """
        导出数据
        
        Args:
            format_type: 导出格式
            
        Returns:
            str: 导出的数据
        """
        data = {
            'contributions': {cid: c.to_dict() for cid, c in self.contributions.items()},
            'contributors': {cid: c.to_dict() for cid, c in self.contributors.items()},
            'reviews': self.reviews,
            'feedback': self.feedback,
            'collaboration_sessions': self.collaboration_sessions,
            'rewards': self.rewards,
            'statistics': self.statistics,
            'exported_at': datetime.now().isoformat()
        }
        
        if format_type == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")

    def import_data(self, data: str, format_type: str = "json") -> bool:
        """
        导入数据
        
        Args:
            data: 数据内容
            format_type: 数据格式
            
        Returns:
            bool: 是否成功
        """
        try:
            if format_type == "json":
                imported_data = json.loads(data)
            else:
                raise ValueError(f"不支持的导入格式: {format_type}")
            
            # 清空现有数据
            self.contributions.clear()
            self.contributors.clear()
            self.reviews.clear()
            self.feedback.clear()
            self.collaboration_sessions.clear()
            self.rewards.clear()
            
            # 导入贡献
            for cid, cdata in imported_data.get('contributions', {}).items():
                contribution = Contribution(
                    contribution_id=cdata['contribution_id'],
                    title=cdata['title'],
                    description=cdata['description'],
                    contribution_type=ContributionType(cdata['contribution_type']),
                    contributor_id=cdata['contributor_id'],
                    created_at=datetime.fromisoformat(cdata['created_at'])
                )
                
                # 设置其他属性
                for key, value in cdata.items():
                    if hasattr(contribution, key):
                        if key in ['created_at', 'updated_at', 'merged_at']:
                            if value:
                                setattr(contribution, key, datetime.fromisoformat(value))
                        elif key == 'status':
                            setattr(contribution, key, ContributionStatus(value))
                        else:
                            setattr(contribution, key, value)
                
                self.contributions[cid] = contribution
            
            # 导入贡献者
            for cid, cdata in imported_data.get('contributors', {}).items():
                contributor = Contributor(
                    contributor_id=cdata['contributor_id'],
                    username=cdata['username'],
                    email=cdata['email'],
                    role=UserRole(cdata['role'])
                )
                
                # 设置其他属性
                for key, value in cdata.items():
                    if hasattr(contributor, key):
                        if key in ['join_date', 'last_active']:
                            setattr(contributor, key, datetime.fromisoformat(value))
                        else:
                            setattr(contributor, key, value)
                
                self.contributors[cid] = contributor
            
            # 导入其他数据
            self.reviews = imported_data.get('reviews', [])
            self.feedback = imported_data.get('feedback', [])
            self.collaboration_sessions = imported_data.get('collaboration_sessions', [])
            self.rewards = imported_data.get('rewards', [])
            self.statistics = imported_data.get('statistics', {})
            
            return True
            
        except Exception as e:
            print(f"导入数据失败: {e}")
            return False