import Observation
import Foundation

enum AnalysisStatus: Equatable, Sendable {
    case ready(score: Int)
    case partial(reason: String)
}

struct AnalysisSummary: Identifiable, Equatable, Sendable {
    let id: UUID
    let title: String
    let timestamp: String
    let thumbnailName: String
    let status: AnalysisStatus
}

@MainActor
@Observable
final class HomeViewModel {
    let recentAnalyses: [AnalysisSummary] = [
        AnalysisSummary(
            id: UUID(uuidString: "45931C2D-852F-4678-B9B6-188B7FA25591")!,
            title: "Morning Carving Session",
            timestamp: "Today, 9:42 AM",
            thumbnailName: "skiHero",
            status: .ready(score: 78)
        ),
        AnalysisSummary(
            id: UUID(uuidString: "1BCE44BC-F4DC-4690-BEA4-B30E693AE7C1")!,
            title: "Blue Run Practice",
            timestamp: "Yesterday, 3:18 PM",
            thumbnailName: "skiHero",
            status: .partial(reason: "More turns needed")
        )
    ]

    func startAnalysis(using router: AppRouter) {
        router.showAnalysisFlow()
    }
}
