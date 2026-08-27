import Observation
import SwiftUI

enum AppTab: Hashable {
    case home
    case history
    case profile
}

enum AppRoute: Hashable {
    case analysisFlow
}

@MainActor
@Observable
final class AppRouter {
    var selectedTab: AppTab = .home
    var homePath: [AppRoute] = []

    func showAnalysisFlow() {
        homePath.append(.analysisFlow)
    }
}

struct RootTabView: View {
    @Bindable var router: AppRouter

    var body: some View {
        Group {
            switch router.selectedTab {
            case .home:
                NavigationStack(path: $router.homePath) {
                    HomeView(router: router)
                        .navigationDestination(for: AppRoute.self) { route in
                            destination(for: route)
                        }
                }
            case .history:
                NavigationStack {
                    PlaceholderTabView(title: "History", systemImage: "clock.arrow.circlepath")
                }
            case .profile:
                NavigationStack {
                    PlaceholderTabView(title: "Profile", systemImage: "person")
                }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            BottomTabBar(selectedTab: $router.selectedTab)
        }
        .animation(.easeInOut(duration: 0.2), value: router.selectedTab)
    }

    @ViewBuilder
    private func destination(for route: AppRoute) -> some View {
        switch route {
        case .analysisFlow:
            AnalysisFlowView()
        }
    }
}

private struct PlaceholderTabView: View {
    let title: String
    let systemImage: String

    var body: some View {
        ContentUnavailableView(title, systemImage: systemImage)
            .navigationTitle(title)
    }
}
