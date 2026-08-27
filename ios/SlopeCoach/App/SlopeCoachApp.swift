import SwiftUI

@main
struct SlopeCoachApp: App {
    @State private var router = AppRouter()

    var body: some Scene {
        WindowGroup {
            RootTabView(router: router)
        }
    }
}
